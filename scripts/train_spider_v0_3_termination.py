#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

from hippocampus import PackConfig
from hippocampus.programs import (
    SyntheticManifoldRenderer,
    default_split_specs_v0_2,
    generate_split_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ActionSchedule,
    build_grouped_development_cases,
    build_model,
    collect_termination_state_dataset,
    evaluate_closed_loop_batches,
    evaluate_termination_state_dataset,
    load_experiment,
    parameter_count,
    train_frozen_null_head,
    train_frozen_termination_head,
    verify_grouped_development_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_SHA256 = (
    "0ed8e27ec44f3773f76b79f1947526f33ba233556b7db91fef04dcb647e5409d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Spider v0.3 heads on frozen controller states."
    )
    parser.add_argument("--evidence-run-dir", type=Path, required=True)
    parser.add_argument("--evidence-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--null-steps", type=int)
    parser.add_argument("--train-cases", type=int, default=0)
    parser.add_argument("--evaluation-cases", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _limit(values, count: int):
    if count < 0:
        raise ValueError("case limits must be non-negative")
    return values if count == 0 else values[:count]


def _resolve_device(experiment, device_name: str | None):
    if device_name is None:
        return experiment
    device = torch.device(device_name)
    training = replace(experiment.training_config)
    return replace(
        experiment,
        device=device,
        training_config=training,
        pack_config=PackConfig(
            device=device,
            value_dtype=experiment.dtype,
        ),
    )


def _pack_cases(
    experiment,
    cases,
    renderer,
    *,
    seed_offset: int,
):
    return tuple(
        pack_rendered_cases(
            (case,),
            (
                renderer.render(
                    case,
                    row_permutation_seed=(
                        experiment.training_config.seed
                        + seed_offset
                        + index
                    ),
                ),
            ),
            schema=experiment.schema,
            pack_config=experiment.pack_config,
        )
        for index, case in enumerate(cases)
    )


def _schedules(raw: dict[str, Any]) -> tuple[ActionSchedule, ...]:
    values = raw["state_collection"]["action_schedules"]
    schedules = tuple(
        ActionSchedule(
            frontier=float(item["frontier"]),
            context=float(item["context"]),
            evidence=float(item["evidence"]),
            termination=float(item["termination"]),
        )
        for item in values
    )
    if not schedules:
        raise ValueError("termination state collection requires schedules")
    return schedules


def _evidence_threshold(metrics: dict[str, Any]) -> float:
    calibration = metrics["calibration"]
    if calibration["split_name"] != "development_calibration":
        raise ValueError("evidence threshold was not calibrated on development")
    return float(calibration["selected"]["raw_probability_threshold"])


def _validate_evidence_run(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("evidence run is missing metrics or checkpoint")
    metrics = _load_json(metrics_path)
    if metrics["dataset_hash"] != DATASET_SHA256:
        raise ValueError("evidence checkpoint uses a different dataset")
    if metrics["sealed_access_count"] != 0:
        raise ValueError("evidence checkpoint accessed sealed data")
    if _sha256(checkpoint_path) != metrics["checkpoint_sha256"]:
        raise ValueError("evidence checkpoint hash mismatch")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("format") != "spider-training-v2":
        raise ValueError("evidence checkpoint is not a Spider training state")
    return metrics, payload, checkpoint_path


def _transfer_evidence_model(
    target,
    source,
    source_state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    source.load_state_dict(source_state, strict=True)
    target_state = target.state_dict()
    copied: list[str] = []
    missing_backbone: list[str] = []
    for name, target_value in target_state.items():
        is_head = name.startswith(
            ("termination_head.", "null_expansion_head.")
        )
        if is_head:
            continue
        source_value = source_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            if not is_head:
                missing_backbone.append(name)
            continue
        target_state[name] = source_value.to(
            device=target_value.device,
            dtype=target_value.dtype,
        )
        copied.append(name)
    if missing_backbone:
        raise ValueError(
            "termination model does not match evidence backbone: "
            + ", ".join(missing_backbone[:8])
        )
    target.load_state_dict(target_state, strict=True)
    return tuple(copied)


def _state_summary(dataset) -> dict[str, Any]:
    factors = dataset.factor_targets
    reason_counts = {
        str(index): int(
            (
                factors.unknown_mask
                & (factors.unknown_reason == index)
            ).sum().item()
        )
        for index in range(4)
    }
    family_counts: dict[str, int] = {}
    schedule_counts: dict[str, int] = {}
    for family in dataset.families:
        family_counts[family] = family_counts.get(family, 0) + 1
    for schedule in dataset.action_schedules:
        schedule_counts[schedule] = schedule_counts.get(schedule, 0) + 1
    return {
        "state_count": dataset.count,
        "evidence_sufficient_positive": int(
            factors.evidence_sufficient.sum().item()
        ),
        "useful_work_positive": int(
            factors.useful_work_remaining.sum().item()
        ),
        "answer_supported_positive": int(
            factors.answer_supported.sum().item()
        ),
        "unknown_reason_counts": reason_counts,
        "family_counts": family_counts,
        "schedule_counts": schedule_counts,
        "exact_stop_count": int(dataset.exact_stop.sum().item()),
        "null_state_count": (
            dataset.null_states.count
            if dataset.null_states is not None
            else 0
        ),
        "null_positive_count": (
            int(dataset.null_states.targets.sum().item())
            if dataset.null_states is not None
            else 0
        ),
    }


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _backbone_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(
            ("termination_head.", "null_expansion_head.")
        )
    }


def _verify_backbone_unchanged(
    model,
    before: dict[str, torch.Tensor],
) -> None:
    for name, expected in before.items():
        actual = model.state_dict()[name].detach().cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(f"frozen evidence parameter changed: {name}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    evidence_metrics, evidence_payload, evidence_checkpoint = (
        _validate_evidence_run(args.evidence_run_dir)
    )
    evidence_experiment = _resolve_device(
        load_experiment(args.evidence_config),
        args.device,
    )
    target_experiment = _resolve_device(
        load_experiment(args.config),
        args.device,
    )
    if (
        evidence_experiment.raw["dataset"]["protocol"]
        != "spider-v0.3-evidence-dev"
        or target_experiment.raw["dataset"]["protocol"]
        != "spider-v0.3-evidence-dev"
    ):
        raise ValueError("termination training requires the v0.3 dev protocol")
    seed = (
        target_experiment.training_config.seed
        if args.seed is None
        else args.seed
    )
    target_experiment = replace(
        target_experiment,
        training_config=replace(
            target_experiment.training_config,
            seed=seed,
        ),
    )
    torch.manual_seed(seed)
    if target_experiment.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA termination run has no visible GPU")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(target_experiment.device)

    source_model = build_model(evidence_experiment)
    target_model = build_model(target_experiment)
    copied_names = _transfer_evidence_model(
        target_model,
        source_model,
        evidence_payload["model"],
    )
    del source_model
    threshold = _evidence_threshold(evidence_metrics)
    controller_config = replace(
        target_experiment.controller_config,
        evidence_threshold=threshold,
    )
    specs = {spec.name: spec for spec in default_split_specs_v0_2()}
    grouped = build_grouped_development_cases(
        generate_split_cases(specs["train"]),
        generate_split_cases(specs["validation_id"]),
    )
    verify_grouped_development_manifest(grouped)
    if grouped.manifest.aggregate_sha256 != DATASET_SHA256:
        raise RuntimeError("termination development dataset hash drift")
    train_cases = _limit(grouped.train, args.train_cases)
    evaluation_cases = _limit(
        grouped.evaluation,
        args.evaluation_cases,
    )
    renderer = SyntheticManifoldRenderer(
        target_experiment.schema,
        query_dim=target_experiment.query_dim,
        seed=91_337,
    )
    train_batches = _pack_cases(
        target_experiment,
        train_cases,
        renderer,
        seed_offset=0,
    )
    evaluation_batches = _pack_cases(
        target_experiment,
        evaluation_cases,
        renderer,
        seed_offset=200_000,
    )
    permuted_evaluation = _pack_cases(
        target_experiment,
        evaluation_cases,
        renderer,
        seed_offset=1_200_000,
    )
    schedules = _schedules(target_experiment.raw)
    collect_null = target_model.null_expansion_head is not None
    training_states = collect_termination_state_dataset(
        target_model,
        train_batches,
        controller_config=controller_config,
        schedules=schedules,
        seed=seed,
        collect_null_states=collect_null,
    )
    evaluation_states = collect_termination_state_dataset(
        target_model,
        evaluation_batches,
        controller_config=controller_config,
        schedules=schedules,
        seed=seed + 100_000,
        collect_null_states=collect_null,
    )
    before = _backbone_snapshot(target_model)
    steps = args.steps or target_experiment.training_config.steps
    termination_training = train_frozen_termination_head(
        target_model,
        training_states,
        steps=steps,
        batch_size=target_experiment.training_config.batch_size,
        learning_rate=target_experiment.training_config.learning_rate,
        seed=seed,
        loss_config=target_experiment.loss_config,
        log_every=target_experiment.training_config.log_every,
    )
    null_training = None
    if collect_null:
        assert training_states.null_states is not None
        null_settings = target_experiment.raw["null_training"]
        null_training = train_frozen_null_head(
            target_model,
            training_states.null_states,
            steps=(
                args.null_steps
                or int(null_settings["steps"])
            ),
            batch_size=int(null_settings["batch_size"]),
            learning_rate=float(null_settings["learning_rate"]),
            seed=seed + 1,
            log_every=int(null_settings["log_every"]),
        )
    _verify_backbone_unchanged(target_model, before)
    state_evaluation = evaluate_termination_state_dataset(
        target_model,
        evaluation_states,
        loss_config=target_experiment.loss_config,
    )
    autonomous = evaluate_closed_loop_batches(
        target_model,
        evaluation_batches,
        split="development_evaluation",
        controller_config=controller_config,
        dataset_version="spider-programs-v0.2",
        evidence_threshold=threshold,
        permuted_batches=permuted_evaluation,
        invariance_sample_limit=min(16, len(evaluation_batches)),
        include_teacher_forced=False,
    )
    autonomous_payload = autonomous.as_dict()
    retention = (
        autonomous.primary_autonomous_success
        / autonomous.fixed_horizon_structural_success
        if autonomous.fixed_horizon_structural_success > 0
        else 0.0
    )
    finite = all(
        math.isfinite(value)
        for value in (
            termination_training.final_loss,
            state_evaluation.loss,
            autonomous.primary_autonomous_success,
            retention,
        )
    )
    invariance_ok = (
        autonomous.invariance["deterministic_replay_mismatches"] == 0
        and autonomous.invariance[
            "row_permutation_decision_mismatches"
        ]
        == 0
    )
    acceptance = {
        "continuation_recall_at_least_0_95": (
            state_evaluation.continuation_recall >= 0.95
        ),
        "premature_stop_rate_below_0_25": (
            state_evaluation.premature_stop_rate < 0.25
        ),
        "autonomous_retention_at_least_0_85": retention >= 0.85,
        "finite": finite,
        "invariance": invariance_ok,
    }
    checkpoint_path = args.output_dir / "checkpoint.pt"
    checkpoint_payload = {
        "format": "spider-termination-v03",
        "model": target_model.state_dict(),
        "model_config": asdict(target_model.config),
        "controller_config": asdict(controller_config),
        "source_evidence_checkpoint_sha256": _sha256(evidence_checkpoint),
        "source_evidence_experiment_id": evidence_metrics["experiment_id"],
        "evidence_threshold": threshold,
        "termination_training": asdict(termination_training),
        "null_training": (
            asdict(null_training) if null_training is not None else None
        ),
        "source_commit": _source_commit(),
        "dataset_hash": DATASET_SHA256,
        "seed": seed,
        "sealed_access_count": 0,
    }
    _atomic_checkpoint(checkpoint_path, checkpoint_payload)
    result = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": (
            "accepted"
            if finite and invariance_ok
            else "guard_violation"
        ),
        "pass": finite and invariance_ok,
        "source_commit": _source_commit(),
        "dataset_version": "spider-programs-v0.2",
        "dataset_hash": DATASET_SHA256,
        "sealed_access_count": 0,
        "config": target_experiment.raw,
        "config_sha256": _sha256(args.config),
        "seed": seed,
        "parameter_count": parameter_count(target_model),
        "copied_evidence_parameter_count": len(copied_names),
        "frozen_backbone_parameter_count": len(before),
        "source_evidence": {
            "experiment_id": evidence_metrics["experiment_id"],
            "source_commit": evidence_metrics["source_commit"],
            "checkpoint_sha256": _sha256(evidence_checkpoint),
            "operating_threshold": threshold,
        },
        "state_datasets": {
            "training": _state_summary(training_states),
            "development_evaluation": _state_summary(
                evaluation_states
            ),
        },
        "termination_training": asdict(termination_training),
        "null_training": (
            asdict(null_training) if null_training is not None else None
        ),
        "state_evaluation": state_evaluation.as_dict(),
        "autonomous_evaluation": autonomous_payload,
        "autonomous_retention": retention,
        "acceptance": acceptance,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(target_experiment.device)
            if target_experiment.device.type == "cuda"
            else 0
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "status": result["status"],
                "continuation_recall": (
                    state_evaluation.continuation_recall
                ),
                "premature_stop_rate": (
                    state_evaluation.premature_stop_rate
                ),
                "primary_autonomous_success": (
                    autonomous.primary_autonomous_success
                ),
                "fixed_horizon_structural_success": (
                    autonomous.fixed_horizon_structural_success
                ),
                "autonomous_retention": retention,
                "checkpoint_sha256": result["checkpoint_sha256"],
            },
            sort_keys=True,
        )
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
