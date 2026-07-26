#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from hippocampus.programs import (
    RECURRENCE_DATASET_VERSION,
    SyntheticManifoldRenderer,
    default_recurrence_necessity_specs,
    generate_recurrence_necessity_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ControllerExecutionPolicy,
    PathStateIntervention,
    build_model,
    calibrate_on_development_batches,
    evaluate_closed_loop_batches,
    load_experiment,
    parameter_count,
    train_oracle_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one fixed-horizon recurrence-necessity experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--train-cases", type=int, default=0)
    parser.add_argument("--eval-cases", type=int, default=0)
    return parser.parse_args()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pack_cases(
    experiment,
    cases,
    renderer,
    *,
    row_seed_offset: int,
):
    return tuple(
        pack_rendered_cases(
            (case,),
            (
                renderer.render(
                    case,
                    row_permutation_seed=(
                        experiment.training_config.seed
                        + row_seed_offset
                        + index
                    ),
                ),
            ),
            schema=experiment.schema,
            pack_config=experiment.pack_config,
        )
        for index, case in enumerate(cases)
    )


def _manifest_hash(name: str) -> str:
    index = json.loads(
        Path("artifacts/spider_v0_2/splits/MANIFEST_INDEX.json").read_text()
    )
    if index["dataset_version"] != RECURRENCE_DATASET_VERSION:
        raise RuntimeError("recurrence dataset manifest version drift")
    if index["sealed_split_present"]:
        raise RuntimeError("recurrence development manifest contains sealed data")
    return str(index["manifests"][name]["sha256"])


def main() -> None:
    args = parse_args()
    for value, name in (
        (args.train_cases, "train-cases"),
        (args.eval_cases, "eval-cases"),
    ):
        if value < 0 or value % 2:
            raise ValueError(f"{name} must be a non-negative even count")
    experiment = load_experiment(args.config)
    if experiment.raw["dataset"]["version"] != RECURRENCE_DATASET_VERSION:
        raise ValueError("training config does not name the recurrence dataset")
    training_config = experiment.training_config
    if args.seed is not None:
        training_config = replace(training_config, seed=args.seed)
    if args.steps is not None:
        training_config = replace(training_config, steps=args.steps)
    experiment = replace(experiment, training_config=training_config)
    if experiment.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no GPU is visible")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    torch.manual_seed(training_config.seed)
    if experiment.device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)
        torch.cuda.reset_peak_memory_stats(experiment.device)

    specs = {
        spec.name: spec for spec in default_recurrence_necessity_specs()
    }
    train_spec = specs["train_recurrence_necessity"]
    validation_spec = specs["validation_recurrence_necessity"]
    train_cases = generate_recurrence_necessity_cases(
        train_spec,
        limit=args.train_cases or None,
    )
    validation_cases = generate_recurrence_necessity_cases(
        validation_spec,
        limit=args.eval_cases or None,
    )
    renderer = SyntheticManifoldRenderer(
        experiment.schema,
        query_dim=experiment.query_dim,
        seed=91_337,
    )
    train_batches = _pack_cases(
        experiment,
        train_cases,
        renderer,
        row_seed_offset=0,
    )
    validation_batches = _pack_cases(
        experiment,
        validation_cases,
        renderer,
        row_seed_offset=100_000,
    )
    permuted_validation = _pack_cases(
        experiment,
        validation_cases,
        renderer,
        row_seed_offset=1_000_000,
    )
    model = build_model(experiment)
    fixed_policy = ControllerExecutionPolicy.oracle_required(
        seed=training_config.seed,
    )
    checkpoint_path = args.output_dir / "checkpoint.pt"
    checkpoint_every = int(
        experiment.raw["training"].get("checkpoint_every", 1000)
    )
    training = train_oracle_batches(
        model,
        train_batches,
        loop_config=training_config,
        loss_config=experiment.loss_config,
        controller_config=experiment.controller_config,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        execution_policy=fixed_policy,
    )
    with (args.output_dir / "history.jsonl").open("w") as handle:
        for record in training.records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    calibration = calibrate_on_development_batches(
        model,
        validation_batches,
        controller_config=experiment.controller_config,
        split_name="validation_recurrence_necessity",
        dataset_version=RECURRENCE_DATASET_VERSION,
        execution_policy=fixed_policy,
    )
    (args.output_dir / "evidence_calibration.json").write_text(
        json.dumps(calibration.as_dict(), indent=2, sort_keys=True) + "\n"
    )
    primary_report = evaluate_closed_loop_batches(
        model,
        validation_batches,
        split="validation_recurrence_necessity",
        controller_config=experiment.controller_config,
        dataset_version=RECURRENCE_DATASET_VERSION,
        evidence_threshold=calibration.threshold,
        permuted_batches=permuted_validation,
        invariance_sample_limit=min(16, len(validation_batches)),
        execution_policy=fixed_policy,
        include_teacher_forced=False,
    )
    state_ablations: dict[str, dict[str, object]] = {}
    if experiment.raw["model"]["kind"] == "spider":
        for intervention in PathStateIntervention:
            policy = ControllerExecutionPolicy.oracle_required(
                intervention=intervention,
                seed=training_config.seed,
            )
            report = evaluate_closed_loop_batches(
                model,
                validation_batches,
                split="validation_recurrence_necessity",
                controller_config=experiment.controller_config,
                dataset_version=RECURRENCE_DATASET_VERSION,
                evidence_threshold=calibration.threshold,
                permuted_batches=permuted_validation,
                invariance_sample_limit=min(8, len(validation_batches)),
                execution_policy=policy,
                include_teacher_forced=False,
            )
            state_ablations[intervention.value] = {
                "structural_success": (
                    report.fixed_horizon_structural_success
                ),
                "final_autonomous_success": (
                    report.primary_autonomous_success
                ),
                "evidence_exact_set_accuracy": report.evidence[
                    "exact_set_accuracy"
                ],
                "evidence_recall": report.evidence["recall"],
                "valid_path_rate": report.rollout[
                    "exact_valid_path_rate"
                ],
                "mean_rounds": report.efficiency["mean_rounds"],
                "replay_mismatches": report.invariance[
                    "deterministic_replay_mismatches"
                ],
                "row_permutation_mismatches": report.invariance[
                    "row_permutation_decision_mismatches"
                ],
            }

    checkpoint_sha = _sha256(checkpoint_path)
    reports = {
        "primary": primary_report.as_dict(),
        "state_ablations": state_ablations,
    }
    finite = all(
        math.isfinite(value)
        for value in (
            primary_report.fixed_horizon_structural_success,
            primary_report.primary_autonomous_success,
            float(primary_report.evidence["f1"]),
        )
    )
    replay_mismatches = int(
        primary_report.invariance["deterministic_replay_mismatches"]
    ) + sum(
        int(report["replay_mismatches"])
        for report in state_ablations.values()
    )
    row_mismatches = int(
        primary_report.invariance["row_permutation_decision_mismatches"]
    ) + sum(
        int(report["row_permutation_mismatches"])
        for report in state_ablations.values()
    )
    result = {
        "experiment_id": args.experiment_id,
        "status": (
            "accepted"
            if finite and not replay_mismatches and not row_mismatches
            else "guard_violation"
        ),
        "pass": finite and not replay_mismatches and not row_mismatches,
        "source_commit": _source_commit(),
        "dataset_version": RECURRENCE_DATASET_VERSION,
        "dataset_hashes": {
            train_spec.name: _manifest_hash(train_spec.name),
            validation_spec.name: _manifest_hash(validation_spec.name),
        },
        "sealed_access_count": 0,
        "config": experiment.raw,
        "resolved_seed": training_config.seed,
        "resolved_steps": training_config.steps,
        "parameter_count": parameter_count(model),
        "training": {
            "runtime_seconds": training.runtime_seconds,
            "initial_metrics": training.initial_metrics.as_dict(),
            "final_metrics": training.final_metrics.as_dict(),
            "action_source_counts": training.action_source_counts,
            "unique_cases_seen": training.unique_cases_seen,
            "training_examples": training.training_examples,
            "history": [asdict(record) for record in training.records],
        },
        "calibration": calibration.as_dict(),
        "reports": reports,
        "primary_structural_success": (
            primary_report.fixed_horizon_structural_success
        ),
        "primary_final_autonomous_success": (
            primary_report.primary_autonomous_success
        ),
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(experiment.device)
            if experiment.device.type == "cuda"
            else 0
        ),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_every": checkpoint_every,
        "guard": {
            "finite": finite,
            "deterministic_replay_mismatches": replay_mismatches,
            "row_permutation_mismatches": row_mismatches,
            "sealed_access_count": 0,
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "checkpoint.manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "source_commit": result["source_commit"],
                "dataset_hashes": result["dataset_hashes"],
                "evidence_threshold": calibration.threshold,
                "sealed_access_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **experiment.raw,
                "training": {
                    **experiment.raw["training"],
                    "seed": training_config.seed,
                    "steps": training_config.steps,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "status": result["status"],
                "structural_success": result[
                    "primary_structural_success"
                ],
                "final_autonomous_success": result[
                    "primary_final_autonomous_success"
                ],
                "runtime_seconds": result["runtime_seconds"],
                "checkpoint_sha256": checkpoint_sha,
            },
            sort_keys=True,
        )
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

