#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from hippocampus.programs import (
    SyntheticManifoldRenderer,
    default_split_specs_v0_2,
    generate_split_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ControllerExecutionPolicy,
    build_grouped_development_cases,
    build_model,
    calibrate_closed_loop_evidence,
    evaluate_closed_loop_batches,
    load_experiment,
    parameter_count,
    train_oracle_batches,
    verify_grouped_development_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one registered Spider v0.3 evidence experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--precision-floor", type=float, default=0.0)
    parser.add_argument("--train-cases", type=int, default=0)
    parser.add_argument("--calibration-cases", type=int, default=0)
    parser.add_argument("--evaluation-cases", type=int, default=0)
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


def _limited(cases, count: int):
    if count < 0:
        raise ValueError("case limits must be non-negative")
    return cases if count == 0 else cases[:count]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse {args.output_dir}")
    experiment = load_experiment(args.config)
    if experiment.raw["dataset"].get("protocol") != (
        "spider-v0.3-evidence-dev"
    ):
        raise ValueError("config does not name the v0.3 evidence protocol")
    training_config = experiment.training_config
    if args.seed is not None:
        training_config = replace(training_config, seed=args.seed)
    experiment = replace(experiment, training_config=training_config)
    stop_after = args.stop_after_steps or training_config.steps
    if experiment.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA experiment requested but no GPU is visible")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    torch.manual_seed(training_config.seed)
    if experiment.device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.seed)
        torch.cuda.reset_peak_memory_stats(experiment.device)

    specs = {spec.name: spec for spec in default_split_specs_v0_2()}
    train_source = generate_split_cases(specs["train"])
    validation_source = generate_split_cases(specs["validation_id"])
    grouped = build_grouped_development_cases(
        train_source,
        validation_source,
    )
    verify_grouped_development_manifest(grouped)
    committed_manifest = json.loads(
        Path(
            "artifacts/spider_v0_3/splits/MANIFEST_INDEX.json"
        ).read_text()
    )
    if (
        grouped.manifest.aggregate_sha256
        != committed_manifest["aggregate_sha256"]
    ):
        raise RuntimeError("v0.3 development manifest hash drift")

    train_cases = _limited(grouped.train, args.train_cases)
    calibration_cases = _limited(
        grouped.calibration,
        args.calibration_cases,
    )
    evaluation_cases = _limited(
        grouped.evaluation,
        args.evaluation_cases,
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
    calibration_batches = _pack_cases(
        experiment,
        calibration_cases,
        renderer,
        row_seed_offset=100_000,
    )
    evaluation_batches = _pack_cases(
        experiment,
        evaluation_cases,
        renderer,
        row_seed_offset=200_000,
    )
    permuted_evaluation = _pack_cases(
        experiment,
        evaluation_cases,
        renderer,
        row_seed_offset=1_200_000,
    )

    model = build_model(experiment)
    fixed_policy = ControllerExecutionPolicy.oracle_required(
        seed=training_config.seed
    )
    checkpoint_path = args.output_dir / "checkpoint.pt"
    training = train_oracle_batches(
        model,
        train_batches,
        loop_config=training_config,
        loss_config=experiment.loss_config,
        controller_config=experiment.controller_config,
        checkpoint_path=checkpoint_path,
        checkpoint_every=int(
            experiment.raw["training"].get("checkpoint_every", 1000)
        ),
        execution_policy=fixed_policy,
        resume_checkpoint=args.resume_checkpoint,
        stop_after_steps=stop_after,
    )
    with (args.output_dir / "history.jsonl").open("w") as handle:
        for record in training.records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    calibration = calibrate_closed_loop_evidence(
        model,
        calibration_batches,
        controller_config=experiment.controller_config,
        split_name="development_calibration",
        dataset_version="spider-programs-v0.2",
        precision_floor=args.precision_floor,
        execution_policy=fixed_policy,
    )
    (args.output_dir / "evidence_calibration.json").write_text(
        json.dumps(calibration.as_dict(), indent=2, sort_keys=True) + "\n"
    )
    report = evaluate_closed_loop_batches(
        model,
        evaluation_batches,
        split="development_evaluation",
        controller_config=experiment.controller_config,
        dataset_version="spider-programs-v0.2",
        evidence_threshold=calibration.threshold,
        permuted_batches=permuted_evaluation,
        invariance_sample_limit=min(16, len(evaluation_batches)),
        execution_policy=fixed_policy,
        include_teacher_forced=False,
    )
    report_payload = report.as_dict()
    (args.output_dir / "development_evaluation.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n"
    )

    finite_values = (
        report.fixed_horizon_structural_success,
        float(report.evidence["precision"]),
        float(report.evidence["recall"]),
        float(report.evidence_pipeline["overall"][
            "exact_evidence_set_accuracy"
        ]),
    )
    replay_mismatches = int(
        report.invariance["deterministic_replay_mismatches"]
    )
    row_mismatches = int(
        report.invariance["row_permutation_decision_mismatches"]
    )
    passed = (
        all(math.isfinite(value) for value in finite_values)
        and replay_mismatches == 0
        and row_mismatches == 0
        and calibration.constraint_satisfied
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    result = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if passed else "guard_violation",
        "pass": passed,
        "source_commit": _source_commit(),
        "dataset_version": "spider-programs-v0.2",
        "development_protocol": grouped.manifest.protocol,
        "dataset_hash": grouped.manifest.aggregate_sha256,
        "dataset_partition_hashes": {
            "train": grouped.manifest.train.sha256,
            "calibration": grouped.manifest.calibration.sha256,
            "evaluation": grouped.manifest.evaluation.sha256,
        },
        "sealed_access_count": 0,
        "config": experiment.raw,
        "resolved_seed": training_config.seed,
        "planned_steps": training_config.steps,
        "completed_steps": training.completed_steps,
        "resumed_from_step": training.resumed_from_step,
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
        "reports": {
            "development_evaluation": report_payload,
        },
        "evidence_gate_metrics": {
            "precision": report.evidence["precision"],
            "recall": report.evidence["recall"],
            "exact_set_accuracy": report.evidence_pipeline["overall"][
                "exact_evidence_set_accuracy"
            ],
            "scored_positive_coverage": report.evidence_pipeline["overall"][
                "scored_positive_coverage"
            ],
            "conditional_selection_recall": report.evidence_pipeline[
                "overall"
            ]["selection_recall_conditioned_on_scored"],
            "false_positives_per_case": report.evidence_pipeline["overall"][
                "false_positives_per_case"
            ],
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(experiment.device)
            if experiment.device.type == "cuda"
            else 0
        ),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "guard": {
            "finite": all(math.isfinite(value) for value in finite_values),
            "calibration_constraint_satisfied": (
                calibration.constraint_satisfied
            ),
            "deterministic_replay_mismatches": replay_mismatches,
            "row_permutation_mismatches": row_mismatches,
            "sealed_access_count": 0,
        },
        "failure_reason": (
            None
            if passed
            else "non-finite metric, calibration, or invariance guard failed"
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "checkpoint.manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "source_commit": result["source_commit"],
                "dataset_hash": result["dataset_hash"],
                "completed_steps": training.completed_steps,
                "planned_steps": training_config.steps,
                "evidence_threshold": calibration.threshold,
                "sealed_access_count": 0,
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
                "completed_steps": training.completed_steps,
                "evidence_gate_metrics": result["evidence_gate_metrics"],
                "checkpoint_sha256": checkpoint_sha256,
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
