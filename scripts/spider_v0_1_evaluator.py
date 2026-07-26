#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from hippocampus.programs import (
    SyntheticManifoldRenderer,
    default_split_specs_v0_2,
    generate_rollout_stress_examples,
    generate_split_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    build_model,
    calibrate_on_development_batches,
    evaluate_closed_loop_batches,
    evaluate_rollout_stress_states,
    load_experiment,
    parameter_count,
    train_oracle_batches,
    validate_v0_1_artifact_input,
    validate_v0_1_split_access,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen Spider v0.1 closed-loop experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cases", type=int, default=0)
    parser.add_argument("--eval-cases", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--diagnostic-checkpoint", type=Path)
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


def _pack_cases(experiment, cases, renderer, *, row_seed_offset: int):
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


def _aggregate_primary(
    reports: dict[str, dict[str, object]],
) -> float:
    total_success = 0.0
    total_cases = 0
    for report in reports.values():
        split = str(report["split"])
        if split == "development_rollout_stress":
            continue
        count = int(report["case_count"])
        total_success += float(report["primary_autonomous_success"]) * count
        total_cases += count
    return total_success / max(1, total_cases)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    experiment = load_experiment(args.config)
    training_config = experiment.training_config
    if args.steps is not None:
        training_config = replace(training_config, steps=args.steps)
    if args.seed is not None:
        training_config = replace(training_config, seed=args.seed)
    experiment = replace(experiment, training_config=training_config)
    if experiment.raw["dataset"]["version"] != "spider-programs-v0.2":
        raise ValueError("v0.1 evaluator requires spider-programs-v0.2")
    if experiment.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA experiment requested but no GPU is visible")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = {spec.name: spec for spec in default_split_specs_v0_2()}
    for spec in specs.values():
        validate_v0_1_split_access(
            spec,
            allow_sealed=False,
        ) if not spec.sealed else None
    split_index = json.loads(
        Path("artifacts/spider_v0_1/splits/MANIFEST_INDEX.json").read_text()
    )
    renderer = SyntheticManifoldRenderer(
        experiment.schema,
        query_dim=experiment.query_dim,
        seed=91_337,
    )
    model = build_model(experiment)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    training_payload: dict[str, object]
    if args.diagnostic_checkpoint is not None:
        validate_v0_1_artifact_input(
            args.diagnostic_checkpoint,
            allow_historical_checkpoint_diagnostic=True,
        )
        payload = torch.load(
            args.diagnostic_checkpoint,
            map_location=experiment.device,
            weights_only=False,
        )
        model.load_state_dict(payload["model"], strict=True)
        training_payload = {
            "diagnostic_only": True,
            "historical_checkpoint": str(args.diagnostic_checkpoint),
            "historical_checkpoint_sha256": _sha256(
                args.diagnostic_checkpoint
            ),
            "steps": 0,
            "unique_cases_seen": 0,
            "training_examples": 0,
            "action_source_counts": {},
        }
        retained_checkpoint = args.diagnostic_checkpoint
    else:
        train_limit = args.train_cases or None
        train_cases = generate_split_cases(
            specs["train"],
            limit=train_limit,
        )
        train_batches = _pack_cases(
            experiment,
            train_cases,
            renderer,
            row_seed_offset=0,
        )
        training = train_oracle_batches(
            model,
            train_batches,
            loop_config=training_config,
            loss_config=experiment.loss_config,
            controller_config=experiment.controller_config,
            checkpoint_path=checkpoint_path,
        )
        training_payload = {
            "diagnostic_only": False,
            "steps": training_config.steps,
            "runtime_seconds": training.runtime_seconds,
            "initial_metrics": training.initial_metrics.as_dict(),
            "final_metrics": training.final_metrics.as_dict(),
            "unique_cases_seen": training.unique_cases_seen,
            "training_examples": training.training_examples,
            "action_source_counts": training.action_source_counts,
            "history": [asdict(record) for record in training.records],
        }
        with (args.output_dir / "history.jsonl").open("w") as handle:
            for record in training.records:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        retained_checkpoint = checkpoint_path

    validation_id_cases = generate_split_cases(
        specs["validation_id"],
        limit=args.eval_cases or None,
    )
    validation_id_batches = _pack_cases(
        experiment,
        validation_id_cases,
        renderer,
        row_seed_offset=10_000,
    )
    calibration = calibrate_on_development_batches(
        model,
        validation_id_batches,
        controller_config=experiment.controller_config,
        split_name="validation_id",
        dataset_version="spider-programs-v0.2",
    )
    (args.output_dir / "evidence_calibration.json").write_text(
        json.dumps(calibration.as_dict(), indent=2, sort_keys=True) + "\n"
    )

    reports: dict[str, dict[str, object]] = {}
    evaluation_names = (
        "validation_id",
        "validation_graph_size_ood",
        "validation_path_length_ood",
        "validation_topology_ood",
        "validation_cardinality_ood",
        "validation_equivalent_view_ood",
        "validation_composition_ood",
        "development_rollout_stress",
    )
    row_mismatches = 0
    replay_mismatches = 0
    peak_memory = 0
    for split_index_value, split_name in enumerate(evaluation_names):
        spec = specs[split_name]
        validate_v0_1_split_access(spec)
        if split_name == "validation_id":
            cases = validation_id_cases
            batches = validation_id_batches
        else:
            cases = generate_split_cases(
                spec,
                limit=args.eval_cases or None,
            )
            batches = _pack_cases(
                experiment,
                cases,
                renderer,
                row_seed_offset=20_000 + split_index_value * 10_000,
            )
        permuted = _pack_cases(
            experiment,
            cases,
            renderer,
            row_seed_offset=200_000 + split_index_value * 10_000,
        )
        report = evaluate_closed_loop_batches(
            model,
            batches,
            split=split_name,
            controller_config=experiment.controller_config,
            evidence_threshold=calibration.threshold,
            permuted_batches=permuted,
            invariance_sample_limit=min(16, len(batches)),
        )
        reports[split_name] = report.as_dict()
        row_mismatches += int(
            report.invariance["row_permutation_decision_mismatches"]
        )
        replay_mismatches += int(
            report.invariance["deterministic_replay_mismatches"]
        )
        peak_memory = max(peak_memory, report.peak_cuda_memory_bytes)

    stress_examples = generate_rollout_stress_examples(
        specs["development_rollout_stress"]
    )
    if args.eval_cases:
        stress_examples = stress_examples[: args.eval_cases]
    stress_cases = tuple(example.case for example in stress_examples)
    stress_batches = _pack_cases(
        experiment,
        stress_cases,
        renderer,
        row_seed_offset=900_000,
    )
    stress_state_report = evaluate_rollout_stress_states(
        model,
        stress_examples,
        stress_batches,
        controller_config=replace(
            experiment.controller_config,
            evidence_threshold=calibration.threshold,
        ),
    )

    primary = _aggregate_primary(reports)
    checkpoint_sha = _sha256(retained_checkpoint)
    result = {
        "pass": (
            row_mismatches == 0
            and replay_mismatches == 0
            and math_is_finite(primary)
        ),
        "status": "accepted",
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_commit": _source_commit(),
        "dataset_version": "spider-programs-v0.2",
        "dataset_split_digest": split_index["aggregate_sha256"],
        "sealed_access_count": 0,
        "score": primary,
        "primary_autonomous_success": primary,
        "config": experiment.raw,
        "parameter_count": parameter_count(model),
        "seed": training_config.seed,
        "device": str(experiment.device),
        "dtype": str(experiment.dtype),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(experiment.device)
            if experiment.device.type == "cuda"
            else None
        ),
        "training": training_payload,
        "calibration": calibration.as_dict(),
        "reports": reports,
        "rollout_stress_states": stress_state_report,
        "row_permutation_decision_mismatches": row_mismatches,
        "deterministic_replay_mismatches": replay_mismatches,
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak_memory,
        "checkpoint_path": str(retained_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "failure_reason": None,
    }
    if not result["pass"]:
        result["status"] = "guard_violation"
        result["failure_reason"] = "non-finite score or invariance guard failed"
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "checkpoint.manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(retained_checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "source_commit": result["source_commit"],
                "dataset_split_digest": result["dataset_split_digest"],
                "evidence_threshold": calibration.threshold,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "status": result["status"],
                "experiment_id": result["experiment_id"],
                "score": result["score"],
                "runtime_seconds": result["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


def math_is_finite(value: float) -> bool:
    return not (value != value or value in {float("inf"), float("-inf")})


if __name__ == "__main__":
    main()
