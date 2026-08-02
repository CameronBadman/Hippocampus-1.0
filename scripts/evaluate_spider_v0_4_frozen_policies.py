#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from hippocampus.spider import (
    EvidenceCandidateObservation,
    EvidencePipelineCaseReport,
    EvidenceRequirement,
    audit_frozen_evidence_policies,
)


ARMS = ("B0", "B1", "B2")
SEEDS = (1701, 1802, 1903)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen Spider v0.4 evidence logits and set policies."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "artifacts/spider_v0_4/phase_b/local_rtx5070ti/runs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spider_v0_4/diagnostics/frozen_policies"),
    )
    return parser.parse_args()


def _report(value: dict[str, object]) -> EvidencePipelineCaseReport:
    requirements = tuple(
        EvidenceRequirement(**item) for item in value["requirements"]
    )
    candidates = tuple(
        EvidenceCandidateObservation(**item)
        for item in value["candidate_observations"]
    )
    return EvidencePipelineCaseReport(
        case_id=str(value["case_id"]),
        family=str(value["family"]),
        horizon=int(value["horizon"]),
        requirements=requirements,
        requirement_observations=(),
        candidate_observations=candidates,
        exact_set_accuracy=float(value["exact_set_accuracy"]),
        true_positives=int(value["true_positives"]),
        false_positives=int(value["false_positives"]),
        false_negatives=int(value["false_negatives"]),
        predicted_cardinality=int(value["predicted_cardinality"]),
        required_cardinality=int(value["required_cardinality"]),
        average_precision=float(value["average_precision"]),
        worst_positive_rank=(
            None
            if value["worst_positive_rank"] is None
            else int(value["worst_positive_rank"])
        ),
        minimum_positive_negative_margin=(
            None
            if value["minimum_positive_negative_margin"] is None
            else float(value["minimum_positive_negative_margin"])
        ),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, int], dict[str, object]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            metrics_path = (
                args.run_root
                / f"V04-phase-B-{arm}-s{seed}"
                / "metrics.json"
            )
            metrics = json.loads(metrics_path.read_text())
            if metrics["sealed_access_count"] != 0:
                raise RuntimeError("frozen policy input records sealed access")
            raw_cases = metrics["development_evaluation"][
                "evidence_pipeline"
            ]["cases"]
            audit = audit_frozen_evidence_policies(
                tuple(_report(case) for case in raw_cases)
            )
            payload = {
                "experiment_id": metrics["experiment_id"],
                "arm": arm,
                "seed": seed,
                "source_commit": metrics["source_commit"],
                "dataset_hash": metrics["dataset_hash"],
                "sealed_access_count": 0,
                **audit.as_dict(),
            }
            results[(arm, seed)] = payload
            (args.output_dir / f"{arm}_s{seed}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )

    arms: dict[str, object] = {}
    for arm in ARMS:
        rows = [results[(arm, seed)] for seed in SEEDS]
        arms[arm] = {
            "mean_global_threshold_exact_set_accuracy": statistics.mean(
                row["overall"]["P0_global_threshold"][
                    "exact_set_accuracy"
                ]
                for row in rows
            ),
            "mean_oracle_cardinality_exact_set_accuracy": statistics.mean(
                row["overall"]["P1_oracle_cardinality"][
                    "exact_set_accuracy"
                ]
                for row in rows
            ),
            "mean_per_case_threshold_exact_set_accuracy": statistics.mean(
                row["overall"]["P2_per_case_threshold"][
                    "exact_set_accuracy"
                ]
                for row in rows
            ),
            "mean_oracle_cardinality_gain": statistics.mean(
                row["oracle_cardinality_exact_set_gain"] for row in rows
            ),
            "seed_branches": [row["recommended_branch"] for row in rows],
        }
    b2_gains = [
        results[("B2", seed)]["oracle_cardinality_exact_set_gain"]
        for seed in SEEDS
    ]
    dataset_hashes = {
        row["dataset_hash"] for row in results.values()
    }
    if len(dataset_hashes) != 1:
        raise RuntimeError("frozen policy inputs use different datasets")
    summary = {
        "dataset_version": "spider-programs-v0.4-aligned-dev",
        "dataset_hash": next(iter(dataset_hashes)),
        "sealed_access_count": 0,
        "arms": arms,
        "decision": {
            "B2_oracle_cardinality_gains": b2_gains,
            "strong_ceiling_seed_count": sum(gain >= 0.15 for gain in b2_gains),
            "set_decoding_branch_indicated": sum(
                gain >= 0.15 for gain in b2_gains
            )
            >= 2,
            "ranking_is_perfect": False,
        },
    }
    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
