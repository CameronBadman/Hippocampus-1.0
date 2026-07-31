#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

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
    build_model,
    evaluate_closed_loop_batches,
    load_experiment,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1701, 1802, 1903)
MODELS = ("pooled", "recurrent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate immutable v0.2 checkpoints with exact evidence tracing."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spider_v0_3/preserved_diagnostics"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )
    parser.add_argument("--case-limit", type=int, default=0)
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


def _run_root(model: str, seed: int) -> Path:
    return (
        ROOT
        / f"artifacts/spider_v0_2/training/isolated/{model}_{seed}"
        / f"REC-{model}-s{seed}-6k/run"
    )


def _pack(experiment, cases, renderer, *, offset: int):
    return tuple(
        pack_rendered_cases(
            (case,),
            (
                renderer.render(
                    case,
                    row_permutation_seed=(
                        experiment.training_config.seed + offset + index
                    ),
                ),
            ),
            schema=experiment.schema,
            pack_config=experiment.pack_config,
        )
        for index, case in enumerate(cases)
    )


def _diagnose(model_name: str, seed: int, cases) -> dict[str, object]:
    config_path = (
        ROOT / f"configs/spider_v0_2/{model_name}_recurrence.json"
    )
    experiment = load_experiment(config_path)
    experiment = replace(
        experiment,
        training_config=replace(
            experiment.training_config,
            seed=seed,
        ),
    )
    run_root = _run_root(model_name, seed)
    historical = json.loads((run_root / "metrics.json").read_text())
    checkpoint = run_root / "checkpoint.pt"
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != historical["checkpoint_sha256"]:
        raise RuntimeError(
            f"immutable checkpoint hash mismatch for {model_name} {seed}"
        )
    if historical["sealed_access_count"] != 0:
        raise RuntimeError("historical diagnostic unexpectedly accessed sealed data")

    renderer = SyntheticManifoldRenderer(
        experiment.schema,
        query_dim=experiment.query_dim,
        seed=91_337,
    )
    batches = _pack(experiment, cases, renderer, offset=100_000)
    permuted = _pack(experiment, cases, renderer, offset=1_000_000)
    model = build_model(experiment)
    payload = torch.load(
        checkpoint,
        map_location=experiment.device,
        weights_only=False,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    policy = ControllerExecutionPolicy.oracle_required(seed=seed)
    report = evaluate_closed_loop_batches(
        model,
        batches,
        split="validation_recurrence_necessity",
        controller_config=experiment.controller_config,
        dataset_version=RECURRENCE_DATASET_VERSION,
        evidence_threshold=float(historical["calibration"]["threshold"]),
        permuted_batches=permuted,
        invariance_sample_limit=min(16, len(batches)),
        execution_policy=policy,
        include_teacher_forced=False,
    )
    overall = report.evidence_pipeline["overall"]
    return {
        "diagnostic_id": f"V03-DIAG-{model_name}-s{seed}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnostic_source_commit": _source_commit(),
        "historical_source_commit": historical["source_commit"],
        "historical_experiment_id": historical["experiment_id"],
        "historical_checkpoint_sha256": checkpoint_hash,
        "historical_evidence_threshold": historical["calibration"][
            "threshold"
        ],
        "dataset_version": RECURRENCE_DATASET_VERSION,
        "dataset_hash": historical["dataset_hashes"][
            "validation_recurrence_necessity"
        ],
        "model": model_name,
        "seed": seed,
        "case_count": len(cases),
        "sealed_access_count": 0,
        "metrics": {
            "reachable_evidence_ceiling": overall[
                "reachable_evidence_ceiling"
            ],
            "scored_positive_coverage": overall[
                "scored_positive_coverage"
            ],
            "conditional_selection_recall": overall[
                "selection_recall_conditioned_on_scored"
            ],
            "recording_recall_conditioned_on_selected": overall[
                "recording_recall_conditioned_on_selected"
            ],
            "evidence_average_precision": overall[
                "micro_evidence_average_precision"
            ],
            "exact_evidence_set_accuracy": overall[
                "exact_evidence_set_accuracy"
            ],
            "false_positives_per_case": overall[
                "false_positives_per_case"
            ],
            "mean_predicted_cardinality": overall[
                "mean_predicted_cardinality"
            ],
            "mean_required_cardinality": overall[
                "mean_required_cardinality"
            ],
            "mean_worst_positive_rank": overall[
                "mean_worst_positive_rank"
            ],
            "minimum_positive_negative_margin": overall[
                "minimum_positive_negative_margin"
            ],
        },
        "evidence_pipeline": report.evidence_pipeline,
        "invariance": report.invariance,
        "execution": report.execution,
    }


def _summary(records: list[dict[str, object]]) -> str:
    lines = [
        "# Preserved Spider v0.2 evidence diagnostics",
        "",
        "These are post-v0.2 measurements. They do not alter the historical "
        "v0.2 result or threshold.",
        "",
        "| Model | Seed | Reachable | Scored | Selected/scored | AP | "
        "Exact set | FP/case | Predicted/required |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            "| {model} | {seed} | {reachable:.4f} | {scored:.4f} | "
            "{selected:.4f} | {ap:.4f} | {exact:.4f} | {fp:.4f} | "
            "{predicted:.3f}/{required:.3f} |".format(
                model=record["model"],
                seed=record["seed"],
                reachable=metrics["reachable_evidence_ceiling"],
                scored=metrics["scored_positive_coverage"],
                selected=metrics["conditional_selection_recall"],
                ap=metrics["evidence_average_precision"],
                exact=metrics["exact_evidence_set_accuracy"],
                fp=metrics["false_positives_per_case"],
                predicted=metrics["mean_predicted_cardinality"],
                required=metrics["mean_required_cardinality"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.case_limit < 0 or args.case_limit % 2:
        raise ValueError("case-limit must be zero or a positive even count")
    spec = next(
        spec
        for spec in default_recurrence_necessity_specs()
        if spec.name == "validation_recurrence_necessity"
    )
    cases = generate_recurrence_necessity_cases(
        spec,
        limit=args.case_limit or None,
    )
    records = [
        _diagnose(model, seed, cases)
        for model in args.models
        for seed in SEEDS
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "diagnostics.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output / "SUMMARY.md").write_text(_summary(records))
    print(_summary(records))


if __name__ == "__main__":
    main()
