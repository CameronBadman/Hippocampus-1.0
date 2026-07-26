"""Aggregate the verified Spider v0.1 isolated Colab replications.

This command is deliberately fail-closed. It emits an experiment ledger and
summary only when all six frozen runs pass deep artifact verification and
their standalone checkpoints and archives are registered in Google Drive.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

try:
    from scripts.verify_spider_v0_1_colab_run import (
        DATASET_DIGEST,
        SOURCE_COMMIT,
        read_json,
        sha256,
        verify_run,
    )
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from verify_spider_v0_1_colab_run import (  # type: ignore[no-redef]
        DATASET_DIGEST,
        SOURCE_COMMIT,
        read_json,
        sha256,
        verify_run,
    )


SEEDS = (1701, 1802, 1903)
MODEL_FAMILIES = ("recurrent", "pooled")
EXPECTED_RUN_IDS = tuple(
    f"L-E00{model_number}-{family}-s{seed}-5k"
    for seed in SEEDS
    for model_number, family in ((4, "recurrent"), (5, "pooled"))
)
VALIDATION_SPLITS = (
    "validation_id",
    "validation_graph_size_ood",
    "validation_path_length_ood",
    "validation_topology_ood",
    "validation_cardinality_ood",
    "validation_equivalent_view_ood",
    "validation_composition_ood",
)
VALIDATION_ID_METRICS = {
    "evidence_exact_set_accuracy": ("evidence", "exact_set_accuracy"),
    "evidence_f1": ("evidence", "f1"),
    "evidence_precision": ("evidence", "precision"),
    "evidence_recall": ("evidence", "recall"),
    "mean_arcs_scored": ("efficiency", "mean_arcs_scored"),
    "mean_contexts_read": ("efficiency", "mean_contexts_read"),
    "mean_rounds": ("efficiency", "mean_rounds"),
    "one_round_stop_rate": ("rollout", "one_round_stop_rate"),
    "risk_among_answered": ("rollout", "risk_among_answered"),
    "termination_accuracy": ("rollout", "termination_accuracy"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def nested_float(value: Mapping[str, Any], path: Sequence[str]) -> float:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    if not isinstance(current, int | float):
        raise TypeError(f"{'.'.join(path)} must be numeric")
    return float(current)


def validate_record(record: Mapping[str, Any]) -> None:
    experiment_id = record.get("experiment_id")
    if experiment_id not in EXPECTED_RUN_IDS:
        raise ValueError(f"unexpected experiment ID: {experiment_id!r}")
    if record.get("status") != "accepted":
        raise RuntimeError(f"{experiment_id}: record is not accepted")
    if record.get("steps") != 5_000:
        raise RuntimeError(f"{experiment_id}: step count drift")
    if record.get("sealed_access_count") != 0:
        raise RuntimeError(f"{experiment_id}: sealed access was reported")
    if record.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError(f"{experiment_id}: source commit drift")
    if record.get("dataset_split_digest") != DATASET_DIGEST:
        raise RuntimeError(f"{experiment_id}: dataset digest drift")

    reports = record.get("reports")
    if not isinstance(reports, Mapping):
        raise TypeError(f"{experiment_id}: reports must be a mapping")
    missing_splits = set(VALIDATION_SPLITS).difference(reports)
    if missing_splits:
        raise RuntimeError(
            f"{experiment_id}: missing validation splits "
            f"{sorted(missing_splits)!r}"
        )
    if any("sealed" in str(name).lower() for name in reports):
        raise RuntimeError(f"{experiment_id}: sealed report was emitted")


def load_records(
    run_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path], dict[str, dict[str, Any]]]:
    run_directories = {
        path.parent.name: path.parent
        for path in run_root.glob("*/*/replication_record.json")
    }
    if set(run_directories) != set(EXPECTED_RUN_IDS):
        raise RuntimeError(
            "isolated run set mismatch: "
            f"{sorted(run_directories)!r}"
        )

    records: list[dict[str, Any]] = []
    verification: dict[str, dict[str, Any]] = {}
    for experiment_id in EXPECTED_RUN_IDS:
        run_directory = run_directories[experiment_id]
        verification[experiment_id] = verify_run(run_directory)
        record = read_json(run_directory / "replication_record.json")
        validate_record(record)
        records.append(record)
    return records, run_directories, verification


def validate_drive_backup(
    backup: Mapping[str, Any],
    run_directories: Mapping[str, Path],
    verification: Mapping[str, Mapping[str, Any]],
) -> None:
    if backup.get("expected_run_count") != len(EXPECTED_RUN_IDS):
        raise RuntimeError("Drive backup expected-run count mismatch")
    if backup.get("sealed_access_allowed") is not False:
        raise RuntimeError("Drive backup does not forbid sealed access")

    uploads = backup.get("uploads")
    if not isinstance(uploads, list):
        raise TypeError("Drive backup uploads must be a list")
    by_experiment = {
        upload.get("experiment_id"): upload
        for upload in uploads
        if isinstance(upload, Mapping)
    }
    if set(by_experiment) != set(EXPECTED_RUN_IDS):
        raise RuntimeError(
            "Drive upload set mismatch: "
            f"{sorted(by_experiment)!r}"
        )
    if len(uploads) != len(by_experiment):
        raise RuntimeError("Drive backup contains duplicate experiment IDs")

    file_ids: set[str] = set()
    for experiment_id in EXPECTED_RUN_IDS:
        upload = by_experiment[experiment_id]
        if upload.get("sealed_access_count") != 0:
            raise RuntimeError(
                f"{experiment_id}: Drive record reports sealed access"
            )
        for kind in ("archive", "checkpoint"):
            file_record = upload.get(kind)
            if not isinstance(file_record, Mapping):
                raise TypeError(
                    f"{experiment_id}: {kind} Drive record is invalid"
                )
            file_id = file_record.get("drive_file_id")
            if not isinstance(file_id, str) or not file_id:
                raise RuntimeError(
                    f"{experiment_id}: {kind} Drive ID is missing"
                )
            if file_id in file_ids:
                raise RuntimeError(f"duplicate Drive file ID: {file_id}")
            file_ids.add(file_id)

        checkpoint_record = upload["checkpoint"]
        verified = verification[experiment_id]
        if checkpoint_record.get("bytes") != verified["checkpoint_bytes"]:
            raise RuntimeError(
                f"{experiment_id}: Drive checkpoint byte count mismatch"
            )
        if checkpoint_record.get("sha256") != verified["checkpoint_sha256"]:
            raise RuntimeError(
                f"{experiment_id}: Drive checkpoint hash mismatch"
            )

        archive_path = (
            run_directories[experiment_id].parent
            / f"{experiment_id}-result.tar.gz"
        )
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        archive_record = upload["archive"]
        if archive_record.get("bytes") != archive_path.stat().st_size:
            raise RuntimeError(
                f"{experiment_id}: Drive archive byte count mismatch"
            )
        if archive_record.get("sha256") != sha256(archive_path):
            raise RuntimeError(
                f"{experiment_id}: Drive archive hash mismatch"
            )


def latest_completion_timestamp(
    run_directories: Mapping[str, Path],
) -> str:
    timestamps: list[str] = []
    for experiment_id in EXPECTED_RUN_IDS:
        job_status = read_json(
            run_directories[experiment_id] / "JOB_STATUS.json"
        )
        if job_status.get("state") not in {"packaging", "finished"}:
            raise RuntimeError(
                f"{experiment_id}: job did not reach packaging"
            )
        if job_status.get("sealed_access_count") != 0:
            raise RuntimeError(
                f"{experiment_id}: job status reports sealed access"
            )
        timestamp = job_status.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp:
            raise RuntimeError(
                f"{experiment_id}: completion timestamp is missing"
            )
        timestamps.append(timestamp)
    return max(timestamps)


def mean_metric(
    records: Iterable[Mapping[str, Any]],
    path: Sequence[str],
) -> float:
    return statistics.fmean(nested_float(record, path) for record in records)


def build_summary(
    records: Sequence[Mapping[str, Any]],
    drive_backup: Mapping[str, Any],
    *,
    completed_at: str | None = None,
) -> dict[str, Any]:
    if {record["experiment_id"] for record in records} != set(
        EXPECTED_RUN_IDS
    ):
        raise RuntimeError("summary input does not contain the frozen run set")

    grouped = {
        family: sorted(
            (
                record
                for record in records
                if record["model_family"] == family
            ),
            key=lambda record: int(record["seed"]),
        )
        for family in MODEL_FAMILIES
    }
    groups: dict[str, Any] = {}
    for family, family_records in grouped.items():
        scores = [float(record["score"]) for record in family_records]
        validation_id_metrics = {
            name: mean_metric(
                family_records,
                ("reports", "validation_id", *path),
            )
            for name, path in VALIDATION_ID_METRICS.items()
        }
        split_primary_means = {
            split: mean_metric(
                family_records,
                ("reports", split, "primary_autonomous_success"),
            )
            for split in VALIDATION_SPLITS
        }
        groups[family] = {
            "mean": statistics.fmean(scores),
            "population_stddev": statistics.pstdev(scores),
            "scores_by_seed": {
                str(record["seed"]): float(record["score"])
                for record in family_records
            },
            "split_primary_means": split_primary_means,
            "validation_id_metrics": validation_id_metrics,
        }

    paired_differences = {
        str(seed): (
            groups["recurrent"]["scores_by_seed"][str(seed)]
            - groups["pooled"]["scores_by_seed"][str(seed)]
        )
        for seed in SEEDS
    }
    recurrent_minus_pooled = (
        groups["recurrent"]["mean"] - groups["pooled"]["mean"]
    )
    winner = "recurrent" if recurrent_minus_pooled > 0 else "pooled"
    return {
        "analysis_status": "post-sealed diagnostic; no selection effect",
        "completed_at": completed_at or utc_now(),
        "dataset_split_digest": DATASET_DIGEST,
        "drive_backup": {
            "folder_id": drive_backup["drive_folder"]["id"],
            "folder_url": drive_backup["drive_folder"]["url"],
            "verified_upload_count": len(drive_backup["uploads"]),
        },
        "groups": groups,
        "paired": {
            "mean_recurrent_minus_pooled": recurrent_minus_pooled,
            "recurrent_minus_pooled_by_seed": paired_differences,
            "recurrent_seed_wins": sum(
                difference > 0 for difference in paired_differences.values()
            ),
            "pooled_seed_wins": sum(
                difference < 0 for difference in paired_differences.values()
            ),
        },
        "primary_metric_winner": winner,
        "run_count": len(records),
        "sealed_access_count": 0,
        "source_model_commit": SOURCE_COMMIT,
        "steps_per_run": 5_000,
        "total_optimizer_steps": 5_000 * len(records),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    groups = summary["groups"]
    lines = [
        "# Spider v0.1 5k A100 replication",
        "",
        "Post-sealed diagnostic only; these runs cannot change model "
        "selection, calibration, or the historical sealed result.",
        "",
        "| Model | Seed scores | Mean | Population SD | ID evidence F1 "
        "| ID one-round stop |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in MODEL_FAMILIES:
        group = groups[family]
        score_text = ", ".join(
            f"{seed}: {score:.4f}"
            for seed, score in group["scores_by_seed"].items()
        )
        metrics = group["validation_id_metrics"]
        lines.append(
            f"| {family} | {score_text} | {group['mean']:.4f} "
            f"| {group['population_stddev']:.4f} "
            f"| {metrics['evidence_f1']:.4f} "
            f"| {metrics['one_round_stop_rate']:.4f} |"
        )

    paired = summary["paired"]
    lines.extend(
        [
            "",
            "## Paired primary-metric result",
            "",
            "| Seed | Recurrent minus pooled |",
            "|---:|---:|",
        ]
    )
    for seed, difference in paired[
        "recurrent_minus_pooled_by_seed"
    ].items():
        lines.append(f"| {seed} | {difference:+.4f} |")
    lines.extend(
        [
            "",
            f"The mean recurrent-minus-pooled difference is "
            f"`{paired['mean_recurrent_minus_pooled']:+.4f}`. "
            f"The post-sealed replication therefore favors "
            f"**{summary['primary_metric_winner']}** on the registered "
            "primary metric. This is a diagnostic result, not a new "
            "selection decision.",
            "",
            "## OOD primary autonomous success",
            "",
            "| Split | Recurrent | Pooled | Difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for split in VALIDATION_SPLITS:
        recurrent = groups["recurrent"]["split_primary_means"][split]
        pooled = groups["pooled"]["split_primary_means"][split]
        lines.append(
            f"| {split} | {recurrent:.4f} | {pooled:.4f} "
            f"| {recurrent - pooled:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"All {summary['run_count']} accepted runs used 5,000 FP32 "
            "optimizer steps on A100 GPUs and reported zero sealed access. "
            "Every archive and standalone checkpoint was hash-verified "
            "locally and registered in "
            f"[Google Drive]({summary['drive_backup']['folder_url']}).",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "colab_5k_experiments.jsonl"
    ledger_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in records
        )
    )
    (output_root / "COLAB_5K_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "COLAB_5K_SUMMARY.md").write_text(
        render_markdown(summary)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/spider_v0_1/colab_5k/isolated"),
    )
    parser.add_argument(
        "--drive-backup",
        type=Path,
        default=Path("artifacts/spider_v0_1/GOOGLE_DRIVE_BACKUP.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/spider_v0_1/colab_5k"),
    )
    args = parser.parse_args()

    records, run_directories, verification = load_records(args.run_root)
    drive_backup = read_json(args.drive_backup)
    validate_drive_backup(drive_backup, run_directories, verification)
    summary = build_summary(
        records,
        drive_backup,
        completed_at=latest_completion_timestamp(run_directories),
    )
    write_outputs(args.output_root, records, summary)
    print(
        json.dumps(
            {
                "drive_uploads_verified": len(drive_backup["uploads"]),
                "primary_metric_winner": summary[
                    "primary_metric_winner"
                ],
                "run_count": summary["run_count"],
                "sealed_access_count": 0,
                "summary": str(
                    args.output_root / "COLAB_5K_SUMMARY.json"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
