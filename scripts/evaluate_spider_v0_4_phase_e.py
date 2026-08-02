#!/usr/bin/env python3
"""Audit set-decoding ceilings from frozen Phase D observations."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from hippocampus.spider import (
    audit_frozen_evidence_policies,
    evidence_pipeline_case_report_from_dict,
)


SEEDS = (1701, 1802, 1903)
DEFAULT_ARMS = ("D0", "D2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen Spider v0.4 Phase D evidence policies."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "artifacts/spider_v0_4/phase_d/local_rtx5070ti/full/runs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spider_v0_4/phase_e/frozen_policies"),
    )
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--primary-arm", default="D0")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _mean_policy(
    rows: list[dict[str, Any]],
    policy: str,
    field: str,
) -> float:
    return statistics.mean(
        float(row["overall"][policy][field]) for row in rows
    )


def main() -> None:
    args = parse_args()
    arms = tuple(str(arm) for arm in args.arms)
    if args.primary_arm not in arms:
        raise ValueError("primary arm must be included in audited arms")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in arms:
        for seed in SEEDS:
            metrics_path = (
                args.run_root
                / f"V04-phase-D-{arm}-full-s{seed}"
                / "metrics.json"
            )
            metrics = _load(metrics_path)
            if metrics["sealed_access_count"] != 0:
                raise RuntimeError("Phase E input records sealed access")
            raw_cases = metrics["development_evaluation"][
                "evidence_pipeline"
            ]["cases"]
            audit = audit_frozen_evidence_policies(
                tuple(
                    evidence_pipeline_case_report_from_dict(case)
                    for case in raw_cases
                )
            )
            payload = {
                "experiment_id": metrics["experiment_id"],
                "arm": arm,
                "seed": seed,
                "source_commit": metrics["source_commit"],
                "selected_checkpoint_sha256": metrics[
                    "selected_checkpoint_sha256"
                ],
                "dataset_hash": metrics["dataset_hash"],
                "development_evaluation_reexecution": False,
                "sealed_access_count": 0,
                **audit.as_dict(),
            }
            results[(arm, seed)] = payload
            _write(args.output_dir / f"{arm}_s{seed}.json", payload)

    dataset_hashes = {
        row["dataset_hash"] for row in results.values()
    }
    if len(dataset_hashes) != 1:
        raise RuntimeError("Phase E inputs use different dataset hashes")
    arm_summaries: dict[str, object] = {}
    for arm in arms:
        rows = [results[(arm, seed)] for seed in SEEDS]
        arm_summaries[arm] = {
            "mean_global_threshold_exact_set_accuracy": _mean_policy(
                rows, "P0_global_threshold", "exact_set_accuracy"
            ),
            "mean_oracle_cardinality_exact_set_accuracy": _mean_policy(
                rows, "P1_oracle_cardinality", "exact_set_accuracy"
            ),
            "mean_per_case_threshold_exact_set_accuracy": _mean_policy(
                rows, "P2_per_case_threshold", "exact_set_accuracy"
            ),
            "mean_oracle_null_exact_set_accuracy": _mean_policy(
                rows, "P3_oracle_null", "exact_set_accuracy"
            ),
            "mean_oracle_cardinality_gain": statistics.mean(
                float(row["oracle_cardinality_exact_set_gain"])
                for row in rows
            ),
            "seed_branches": [
                str(row["recommended_branch"]) for row in rows
            ],
        }

    primary_gains = [
        float(
            results[(args.primary_arm, seed)][
                "oracle_cardinality_exact_set_gain"
            ]
        )
        for seed in SEEDS
    ]
    strong_seed_count = sum(gain >= 0.15 for gain in primary_gains)
    branch = (
        "set_decoding"
        if strong_seed_count >= 2
        else "ranking_and_hard_negatives"
    )
    summary = {
        "campaign": "Spider v0.4 Phase E frozen-logit ceiling",
        "dataset_version": "spider-programs-v0.4.1-aligned-evidence-dev",
        "dataset_hash": next(iter(dataset_hashes)),
        "primary_arm": args.primary_arm,
        "development_evaluation_reexecution": False,
        "sealed_access_count": 0,
        "arms": arm_summaries,
        "decision": {
            "primary_oracle_cardinality_gains": primary_gains,
            "strong_ceiling_seed_count": strong_seed_count,
            "selected_branch": branch,
            "run_set_decoding_first": branch == "set_decoding",
            "run_ranking_first": branch == "ranking_and_hard_negatives",
        },
    }
    _write(args.output_dir / "SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
