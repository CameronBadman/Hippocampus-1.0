#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from hippocampus.programs import (
    FreshRenderedBatchSource,
    SyntheticManifoldRenderer,
    default_aligned_dev_specs,
    generate_aligned_dev_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ControllerExecutionPolicy,
    build_model,
    evaluate_closed_loop_batches,
    fast_calibrate_closed_loop_evidence,
    load_experiment,
    parameter_count,
    train_oracle_batches,
)


ROOT = Path(__file__).resolve().parents[1]
PRECISION_FLOOR = 0.90
COVERAGE_FLOOR = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one preregistered Spider v0.4 renderer causal arm."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-cases", type=int, default=512)
    parser.add_argument("--selection-cases", type=int, default=512)
    parser.add_argument("--calibration-cases", type=int, default=512)
    parser.add_argument("--evaluation-cases", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _pack_static(
    cases,
    *,
    renderer,
    experiment,
    row_seed_offset: int,
):
    return tuple(
        pack_rendered_cases(
            (case,),
            (
                renderer.render(
                    case,
                    row_permutation_seed=(row_seed_offset + index),
                ),
            ),
            schema=experiment.schema,
            pack_config=experiment.pack_config,
        )
        for index, case in enumerate(cases)
    )


def _load_model_checkpoint(experiment, checkpoint: Path):
    model = build_model(experiment)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def _selection_key(report: dict[str, Any], step: int) -> tuple[float, ...]:
    selected = report["calibration"]["selected"]
    overall = report["reference_pipeline"]["overall"]
    worst = overall["mean_worst_positive_rank"]
    return (
        float(report["calibration"]["constraint_satisfied"]),
        float(selected["exact_set_accuracy"]),
        float(selected["recall"]),
        float(overall["macro_evidence_average_precision"]),
        -float(selected["false_positives_per_case"]),
        -float("inf") if worst is None else -float(worst),
        -float(step),
    )


def _family_metrics(pipeline: dict[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for family, values in pipeline["by_family"].items():
        true_positive = int(values["true_positives"])
        false_positive = int(values["false_positives"])
        false_negative = int(values["false_negatives"])
        result[family] = {
            **values,
            "precision": true_positive
            / max(1, true_positive + false_positive),
            "recall": true_positive
            / max(1, true_positive + false_negative),
        }
    return result


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    source_commit = _source_commit()
    experiment = load_experiment(args.config)
    if experiment.raw["dataset"].get("protocol") != (
        "spider-v0.4-renderer-causal"
    ):
        raise ValueError("config does not name the v0.4 renderer protocol")
    manifest_path = ROOT / experiment.raw["dataset"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    expected_hash = experiment.raw["dataset"]["aggregate_sha256"]
    if manifest["aggregate_sha256"] != expected_hash:
        raise RuntimeError("v0.4 partition manifest hash drift")
    if manifest["sealed_access_count"] != 0:
        raise RuntimeError("v0.4 development manifest records sealed access")
    if experiment.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered Phase B runs require the local CUDA GPU")
    training_config = replace(
        experiment.training_config,
        seed=args.seed,
        steps=args.steps,
    )
    experiment = replace(experiment, training_config=training_config)
    full_protocol = (
        args.train_cases == 512
        and args.selection_cases == 512
        and args.calibration_cases == 512
        and args.evaluation_cases == 1024
        and args.steps == 1000
    )
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(experiment.device)

    specs = {spec.name: spec for spec in default_aligned_dev_specs()}
    train_cases = generate_aligned_dev_cases(
        specs["training"], limit=args.train_cases
    )
    selection_cases = generate_aligned_dev_cases(
        specs["model_selection"], limit=args.selection_cases
    )
    calibration_cases = generate_aligned_dev_cases(
        specs["calibration"], limit=args.calibration_cases
    )
    evaluation_cases = generate_aligned_dev_cases(
        specs["development_evaluation"], limit=args.evaluation_cases
    )
    renderer_data = experiment.raw["renderer"]
    renderer = SyntheticManifoldRenderer(
        experiment.schema,
        query_dim=experiment.query_dim,
        seed=int(renderer_data["seed"]),
        dtype=experiment.dtype,
        geometry=str(renderer_data["geometry"]),
    )
    fixed_policy = ControllerExecutionPolicy.oracle_required(seed=args.seed)
    checkpoint_records: list[dict[str, object]] = []
    training_payload: dict[str, object] | None = None
    historical_template = experiment.raw.get("historical_checkpoint_template")
    if historical_template is not None:
        checkpoint_paths = [
            ROOT / str(historical_template).format(seed=args.seed)
        ]
        if not checkpoint_paths[0].is_file():
            raise FileNotFoundError(checkpoint_paths[0])
    else:
        model = build_model(experiment)
        train_source = FreshRenderedBatchSource(
            train_cases,
            renderer=renderer,
            schema=experiment.schema,
            base_row_seed=args.seed * 10_000,
            pack_config=experiment.pack_config,
        )
        monitor = _pack_static(
            train_cases[:8],
            renderer=renderer,
            experiment=experiment,
            row_seed_offset=700_000 + args.seed,
        )
        checkpoint_path = args.output_dir / "checkpoint.pt"
        training = train_oracle_batches(
            model,
            train_source,
            loop_config=training_config,
            loss_config=experiment.loss_config,
            controller_config=experiment.controller_config,
            checkpoint_path=checkpoint_path,
            checkpoint_every=250,
            execution_policy=fixed_policy,
            monitor_batches=monitor,
        )
        training_payload = {
            "runtime_seconds": training.runtime_seconds,
            "initial_metrics": training.initial_metrics.as_dict(),
            "final_metrics": training.final_metrics.as_dict(),
            "action_source_counts": training.action_source_counts,
            "unique_cases_seen": training.unique_cases_seen,
            "training_examples": training.training_examples,
            "presentation_count_total": sum(
                train_source.presentation_counts
            ),
            "history": [asdict(record) for record in training.records],
        }
        _write_json(args.output_dir / "training.json", training_payload)
        checkpoint_paths = [
            args.output_dir / f"checkpoint_step_{step:06d}.pt"
            for step in (250, 500, 750)
            if step < args.steps
        ] + [checkpoint_path]
        del model, train_source, monitor
        _release_cuda_cache()

    selection_batches = _pack_static(
        selection_cases,
        renderer=renderer,
        experiment=experiment,
        row_seed_offset=1_000_000 + args.seed,
    )
    for checkpoint_path in checkpoint_paths:
        model, payload = _load_model_checkpoint(experiment, checkpoint_path)
        step = int(payload["step"])
        selection = fast_calibrate_closed_loop_evidence(
            model,
            selection_batches,
            controller_config=experiment.controller_config,
            split_name="model_selection",
            dataset_version="spider-programs-v0.4-aligned-dev",
            precision_floor=PRECISION_FLOOR,
            coverage_floor=COVERAGE_FLOOR,
            execution_policy=fixed_policy,
            fit_temperature=False,
            exact_candidate_count=2,
        )
        record = {
            "step": step,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            **selection.as_dict(),
        }
        checkpoint_records.append(record)
        _write_json(
            args.output_dir / "model_selection" / f"step_{step:06d}.json",
            record,
        )
        del model, payload, selection
        _release_cuda_cache()
    selected_record = max(
        checkpoint_records,
        key=lambda record: _selection_key(record, int(record["step"])),
    )
    selected_checkpoint = Path(str(selected_record["checkpoint_path"]))
    del selection_batches
    _release_cuda_cache()

    model, _ = _load_model_checkpoint(
        experiment,
        selected_checkpoint,
    )
    calibration_batches = _pack_static(
        calibration_cases,
        renderer=renderer,
        experiment=experiment,
        row_seed_offset=2_000_000 + args.seed,
    )
    calibration = fast_calibrate_closed_loop_evidence(
        model,
        calibration_batches,
        controller_config=experiment.controller_config,
        split_name="calibration",
        dataset_version="spider-programs-v0.4-aligned-dev",
        precision_floor=PRECISION_FLOOR,
        coverage_floor=COVERAGE_FLOOR,
        execution_policy=fixed_policy,
        fit_temperature=True,
        exact_candidate_count=3,
    )
    _write_json(args.output_dir / "calibration.json", calibration.as_dict())
    del calibration_batches
    _release_cuda_cache()

    evaluation_batches = _pack_static(
        evaluation_cases,
        renderer=renderer,
        experiment=experiment,
        row_seed_offset=3_000_000 + args.seed,
    )
    permuted_batches = _pack_static(
        evaluation_cases,
        renderer=renderer,
        experiment=experiment,
        row_seed_offset=4_000_000 + args.seed,
    )
    evaluation = evaluate_closed_loop_batches(
        model,
        evaluation_batches,
        split="development_evaluation",
        controller_config=experiment.controller_config,
        dataset_version="spider-programs-v0.4-aligned-dev",
        evidence_threshold=calibration.calibration.threshold,
        permuted_batches=permuted_batches,
        invariance_sample_limit=min(32, len(evaluation_batches)),
        execution_policy=fixed_policy,
        include_teacher_forced=False,
    )
    evaluation_payload = evaluation.as_dict()
    _write_json(
        args.output_dir / "development_evaluation.json",
        evaluation_payload,
    )
    overall = evaluation.evidence_pipeline["overall"]
    family_metrics = _family_metrics(evaluation.evidence_pipeline)
    finite = all(
        math.isfinite(float(value))
        for value in (
            overall["exact_evidence_set_accuracy"],
            evaluation.evidence["precision"],
            evaluation.evidence["recall"],
            overall["scored_positive_coverage"],
            overall["macro_evidence_average_precision"],
        )
    )
    replay_mismatches = int(
        evaluation.invariance["deterministic_replay_mismatches"]
    )
    row_mismatches = int(
        evaluation.invariance["row_permutation_decision_mismatches"]
    )
    mechanically_valid = finite and replay_mismatches == 0 and row_mismatches == 0
    primary_constraint = (
        float(evaluation.evidence["precision"]) >= PRECISION_FLOOR
        and float(overall["scored_positive_coverage"]) >= COVERAGE_FLOOR
    )
    result = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": (
            "accepted" if mechanically_valid and full_protocol else
            "smoke" if mechanically_valid else
            "guard_violation"
        ),
        "pass": mechanically_valid,
        "full_protocol": full_protocol,
        "source_commit": source_commit,
        "config_path": str(args.config),
        "config_sha256": _sha256(args.config),
        "dataset_version": manifest["dataset_version"],
        "dataset_hash": manifest["aggregate_sha256"],
        "sealed_access_count": 0,
        "resolved_seed": args.seed,
        "renderer": renderer_data,
        "parameter_count": parameter_count(model),
        "planned_steps": args.steps,
        "selected_step": int(selected_record["step"]),
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": _sha256(selected_checkpoint),
        "model_selection": checkpoint_records,
        "calibration": calibration.as_dict(),
        "training": training_payload,
        "development_evaluation": evaluation_payload,
        "primary_metric": {
            "exact_evidence_set_accuracy": overall[
                "exact_evidence_set_accuracy"
            ],
            "precision": evaluation.evidence["precision"],
            "recall": evaluation.evidence["recall"],
            "scored_positive_coverage": overall[
                "scored_positive_coverage"
            ],
            "macro_average_precision": overall[
                "macro_evidence_average_precision"
            ],
            "false_positives_per_case": overall[
                "false_positives_per_case"
            ],
            "mean_worst_positive_rank": overall[
                "mean_worst_positive_rank"
            ],
            "constraint_satisfied": primary_constraint,
        },
        "per_family": family_metrics,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(
            experiment.device
        ),
        "guards": {
            "finite": finite,
            "deterministic_replay_mismatches": replay_mismatches,
            "row_permutation_decision_mismatches": row_mismatches,
            "sealed_access_count": 0,
        },
    }
    _write_json(args.output_dir / "metrics.json", result)
    _write_json(
        args.output_dir / "checkpoint.manifest.json",
        {
            "source_commit": source_commit,
            "dataset_hash": manifest["aggregate_sha256"],
            "selected_step": int(selected_record["step"]),
            "checkpoint_path": str(selected_checkpoint),
            "checkpoint_sha256": _sha256(selected_checkpoint),
            "evidence_threshold": calibration.calibration.threshold,
            "temperature": asdict(calibration.calibration.temperature),
            "sealed_access_count": 0,
        },
    )
    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "status": result["status"],
                "selected_step": result["selected_step"],
                "primary_metric": result["primary_metric"],
            },
            sort_keys=True,
        )
    )
    if not mechanically_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
