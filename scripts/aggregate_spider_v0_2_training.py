#!/usr/bin/env python3
"""Aggregate the frozen Spider v0.2 recurrence training matrix.

The command is intentionally fail-closed. It accepts exactly three paired
recurrent/pooled seeds, rejects invariant or sealed-access violations, and
applies the decision rules frozen before training. Fixed-horizon structural
success is the primary metric; learned termination is diagnostic only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from typing import Any


SOURCE_COMMIT = "acb533666d481daf9b6fb56562d69a5dd78c5e0e"
DATASET_VERSION = "spider-programs-v0.3-recurrence-dev"
TRAIN_MANIFEST_SHA256 = (
    "ff36529a8090581f6156a8fc36258e4a14eee9a542955623b70550001469fe56"
)
VALIDATION_MANIFEST_SHA256 = (
    "67c2273e4899af179bc1e10185742b806d751f5f5dba858c771f2eca8a6af4aa"
)
SEEDS = (1701, 1802, 1903)
MODELS = ("recurrent", "pooled")
EXPECTED_RUN_IDS = tuple(
    f"REC-{model}-s{seed}-6k"
    for model in MODELS
    for seed in SEEDS
)
STATE_INTERVENTIONS = (
    "none",
    "reset",
    "detach",
    "shuffle",
    "pooled_current_node",
)
CAUSAL_STATE_INTERVENTIONS = ("reset", "shuffle")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_run(run_directory: Path) -> dict[str, Any]:
    path = Path(__file__).with_name(
        "verify_spider_v0_2_recurrence_run.py"
    )
    spec = importlib.util.spec_from_file_location(
        "spider_v02_recurrence_verifier",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load recurrence verifier from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_run(run_directory)


def _numeric(value: object, *, field: str) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"{field} must be finite")
    return resolved


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _nested_float(
    value: Mapping[str, Any],
    path: Sequence[str],
) -> float:
    current: object = value
    for key in path:
        current = _mapping(current, field=".".join(path)).get(key)
    return _numeric(current, field=".".join(path))


def validate_record(record: Mapping[str, Any]) -> None:
    """Validate one enriched, deeply verified experiment record."""

    experiment_id = record.get("experiment_id")
    if experiment_id not in EXPECTED_RUN_IDS:
        raise ValueError(f"unexpected experiment ID: {experiment_id!r}")
    if record.get("sealed_access_count") != 0:
        raise RuntimeError(f"{experiment_id}: sealed access was reported")
    if record.get("status") != "accepted":
        raise RuntimeError(f"{experiment_id}: run is not accepted")
    if record.get("failure_reason") is not None:
        raise RuntimeError(f"{experiment_id}: accepted run has a failure")
    if record.get("steps") != 6_000:
        raise RuntimeError(f"{experiment_id}: training step count drift")
    if record.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError(f"{experiment_id}: source commit drift")

    model = record.get("model")
    seed = record.get("seed")
    if model not in MODELS or seed not in SEEDS:
        raise RuntimeError(f"{experiment_id}: model/seed is invalid")
    if experiment_id != f"REC-{model}-s{seed}-6k":
        raise RuntimeError(f"{experiment_id}: model/seed label drift")
    if record.get("dataset_version") not in (None, DATASET_VERSION):
        raise RuntimeError(f"{experiment_id}: dataset version drift")
    hashes = record.get("dataset_hashes")
    if hashes is not None:
        hashes = _mapping(hashes, field="dataset_hashes")
        if (
            hashes.get("train_recurrence_necessity")
            != TRAIN_MANIFEST_SHA256
        ):
            raise RuntimeError(f"{experiment_id}: train manifest drift")
        if (
            hashes.get("validation_recurrence_necessity")
            != VALIDATION_MANIFEST_SHA256
        ):
            raise RuntimeError(
                f"{experiment_id}: validation manifest drift"
            )

    reports = _mapping(record.get("reports"), field="reports")
    primary = _mapping(reports.get("primary"), field="reports.primary")
    if int(primary.get("case_count", -1)) != 128:
        raise RuntimeError(f"{experiment_id}: validation case-count drift")
    invariance = _mapping(
        primary.get("invariance"),
        field="reports.primary.invariance",
    )
    for field in (
        "deterministic_replay_mismatches",
        "row_permutation_decision_mismatches",
    ):
        if int(invariance.get(field, -1)) != 0:
            raise RuntimeError(f"{experiment_id}: {field} is nonzero")

    primary_structural = _numeric(
        record.get("primary_structural_success"),
        field="primary_structural_success",
    )
    report_structural = primary.get("fixed_horizon_structural_success")
    if report_structural is not None and not math.isclose(
        primary_structural,
        _numeric(
            report_structural,
            field="reports.primary.fixed_horizon_structural_success",
        ),
        abs_tol=0.0,
    ):
        raise RuntimeError(f"{experiment_id}: primary metric drift")

    state_ablations = _mapping(
        reports.get("state_ablations"),
        field="reports.state_ablations",
    )
    expected_interventions = (
        set(STATE_INTERVENTIONS) if model == "recurrent" else set()
    )
    if set(state_ablations) != expected_interventions:
        raise RuntimeError(
            f"{experiment_id}: state-ablation set drift "
            f"{sorted(state_ablations)!r}"
        )
    for intervention, report in state_ablations.items():
        report = _mapping(
            report,
            field=f"state_ablations.{intervention}",
        )
        for field in ("replay_mismatches", "row_permutation_mismatches"):
            if int(report.get(field, -1)) != 0:
                raise RuntimeError(
                    f"{experiment_id}: {intervention} {field} is nonzero"
                )
        _numeric(
            report.get("structural_success"),
            field=f"state_ablations.{intervention}.structural_success",
        )


def state_use_decision(
    degradations: Mapping[str, Mapping[int, float]],
) -> dict[str, Any]:
    """Apply the pre-registered material-state-use rule.

    Detach is a training-gradient intervention and is forward-equivalent during
    evaluation. Pooled-current-node remains descriptive. Only reset and
    graph-local shuffling are causal forward state-removal tests in the frozen
    rule.
    """

    missing = set(STATE_INTERVENTIONS).difference(
        {"none"}, degradations
    )
    if missing:
        raise RuntimeError(
            f"state degradation set is incomplete: {sorted(missing)!r}"
        )
    by_intervention: dict[str, dict[str, Any]] = {}
    for intervention in degradations:
        by_seed = degradations[intervention]
        if set(by_seed) != set(SEEDS):
            raise RuntimeError(
                f"{intervention}: state-degradation seed set drift"
            )
        values = [_numeric(by_seed[seed], field=intervention) for seed in SEEDS]
        mean = statistics.fmean(values)
        count = sum(value >= 0.05 for value in values)
        by_intervention[intervention] = {
            "mean_degradation": mean,
            "seed_degradations": {
                str(seed): float(by_seed[seed]) for seed in SEEDS
            },
            "seeds_at_or_above_0_05": count,
            "causal_forward_ablation": (
                intervention in CAUSAL_STATE_INTERVENTIONS
            ),
            "material": (
                intervention in CAUSAL_STATE_INTERVENTIONS
                and mean >= 0.05
                and count >= 2
            ),
        }
    return {
        "by_intervention": by_intervention,
        "material_state_use": any(
            by_intervention[name]["material"]
            for name in CAUSAL_STATE_INTERVENTIONS
        ),
    }


def _group_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    structural = [
        _numeric(
            record["primary_structural_success"],
            field="primary_structural_success",
        )
        for record in records
    ]
    final = [
        _numeric(
            record["primary_final_autonomous_success"],
            field="primary_final_autonomous_success",
        )
        for record in records
    ]
    return {
        "structural_mean": statistics.fmean(structural),
        "structural_population_stddev": statistics.pstdev(structural),
        "structural_by_seed": {
            str(record["seed"]): float(record["primary_structural_success"])
            for record in records
        },
        "final_state_mean": statistics.fmean(final),
        "final_state_by_seed": {
            str(record["seed"]): float(
                record["primary_final_autonomous_success"]
            )
            for record in records
        },
        "evidence_exact_set_accuracy_mean": statistics.fmean(
            _nested_float(
                _mapping(record["reports"], field="reports"),
                ("primary", "evidence", "exact_set_accuracy"),
            )
            for record in records
        ),
        "evidence_recall_mean": statistics.fmean(
            _nested_float(
                _mapping(record["reports"], field="reports"),
                ("primary", "evidence", "recall"),
            )
            for record in records
        ),
        "valid_path_rate_mean": statistics.fmean(
            _nested_float(
                _mapping(record["reports"], field="reports"),
                ("primary", "rollout", "exact_valid_path_rate"),
            )
            for record in records
        ),
        "mean_rounds": statistics.fmean(
            _nested_float(
                _mapping(record["reports"], field="reports"),
                ("primary", "efficiency", "mean_rounds"),
            )
            for record in records
        ),
        "mean_arcs_scored": statistics.fmean(
            _nested_float(
                _mapping(record["reports"], field="reports"),
                ("primary", "efficiency", "mean_arcs_scored"),
            )
            for record in records
        ),
    }


def build_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Build the paired scientific result from the exact frozen run set."""

    by_id = {str(record.get("experiment_id")): record for record in records}
    if set(by_id) != set(EXPECTED_RUN_IDS) or len(by_id) != len(records):
        raise RuntimeError("summary input does not contain the frozen run set")
    for record in records:
        validate_record(record)

    grouped = {
        model: sorted(
            (
                record
                for record in records
                if record["model"] == model
            ),
            key=lambda record: int(record["seed"]),
        )
        for model in MODELS
    }
    groups = {
        model: _group_metrics(model_records)
        for model, model_records in grouped.items()
    }
    paired_by_seed: dict[str, dict[str, float]] = {}
    structural_deltas: list[float] = []
    final_deltas: list[float] = []
    for seed in SEEDS:
        recurrent = groups["recurrent"]["structural_by_seed"][str(seed)]
        pooled = groups["pooled"]["structural_by_seed"][str(seed)]
        recurrent_final = groups["recurrent"]["final_state_by_seed"][str(seed)]
        pooled_final = groups["pooled"]["final_state_by_seed"][str(seed)]
        structural_delta = recurrent - pooled
        final_delta = recurrent_final - pooled_final
        structural_deltas.append(structural_delta)
        final_deltas.append(final_delta)
        paired_by_seed[str(seed)] = {
            "recurrent_structural": recurrent,
            "pooled_structural": pooled,
            "structural_delta": structural_delta,
            "recurrent_final_state": recurrent_final,
            "pooled_final_state": pooled_final,
            "final_state_delta": final_delta,
        }

    recurrent_records = {
        int(record["seed"]): record for record in grouped["recurrent"]
    }
    degradations: dict[str, dict[int, float]] = {
        intervention: {} for intervention in STATE_INTERVENTIONS if intervention != "none"
    }
    state_scores: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        reports = _mapping(
            recurrent_records[seed]["reports"],
            field="reports",
        )
        state = _mapping(
            reports["state_ablations"],
            field="state_ablations",
        )
        intact = _nested_float(
            state,
            ("none", "structural_success"),
        )
        state_scores[str(seed)] = {"none": intact}
        for intervention in degradations:
            score = _nested_float(
                state,
                (intervention, "structural_success"),
            )
            state_scores[str(seed)][intervention] = score
            degradations[intervention][seed] = intact - score
    state_use = state_use_decision(degradations)
    state_use["scores_by_seed"] = state_scores

    mean_delta = statistics.fmean(structural_deltas)
    recurrent_wins = sum(delta > 0 for delta in structural_deltas)
    replay_mismatches = sum(
        int(
            _mapping(
                _mapping(record["reports"], field="reports")["primary"],
                field="reports.primary",
            )["invariance"]["deterministic_replay_mismatches"]
        )
        for record in records
    )
    row_mismatches = sum(
        int(
            _mapping(
                _mapping(record["reports"], field="reports")["primary"],
                field="reports.primary",
            )["invariance"]["row_permutation_decision_mismatches"]
        )
        for record in records
    )
    return {
        "analysis_status": (
            "post-sealed architectural diagnostic; no selection effect"
        ),
        "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
        "dataset_version": DATASET_VERSION,
        "groups": groups,
        "paired": {
            "by_seed": paired_by_seed,
            "mean_structural_delta": mean_delta,
            "population_stddev_structural_delta": statistics.pstdev(
                structural_deltas
            ),
            "mean_final_state_delta": statistics.fmean(final_deltas),
            "recurrent_seed_wins": recurrent_wins,
            "pooled_seed_wins": sum(delta < 0 for delta in structural_deltas),
            "ties": sum(delta == 0 for delta in structural_deltas),
        },
        "state_use": state_use,
        "decision": {
            "recurrent_advantage": (
                mean_delta >= 0.02 and recurrent_wins >= 2
            ),
            "required_mean_paired_delta": 0.02,
            "required_seed_wins": 2,
            "material_state_use": state_use["material_state_use"],
        },
        "guards": {
            "deterministic_replay_mismatches": replay_mismatches,
            "row_permutation_mismatches": row_mismatches,
            "sealed_access_count": 0,
        },
        "run_count": len(records),
        "sealed_access_count": 0,
        "source_commit": SOURCE_COMMIT,
        "steps_per_run": 6_000,
        "total_optimizer_steps": 6_000 * len(records),
        "total_runtime_seconds": sum(
            _numeric(record["runtime_seconds"], field="runtime_seconds")
            for record in records
        ),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    paired = summary["paired"]
    groups = summary["groups"]
    lines = [
        "# Spider v0.2 Fixed-Horizon A100 Comparison",
        "",
        "Post-sealed architectural diagnostic only. Learned stopping is "
        "suppressed during the registered comparison, and no historical or "
        "new sealed set is opened.",
        "",
        "| Seed | Recurrent structural | Pooled structural | R − P |",
        "|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = paired["by_seed"][str(seed)]
        lines.append(
            f"| {seed} | {row['recurrent_structural']:.4f} | "
            f"{row['pooled_structural']:.4f} | "
            f"{row['structural_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Mean recurrent-minus-pooled structural delta: "
            f"**{paired['mean_structural_delta']:+.4f}**; recurrent seed "
            f"wins: **{paired['recurrent_seed_wins']}/3**.",
            "",
            "| Model | Structural mean | Evidence exact | Evidence recall | "
            "Valid path | Mean rounds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        group = groups[model]
        lines.append(
            f"| {model} | {group['structural_mean']:.4f} | "
            f"{group['evidence_exact_set_accuracy_mean']:.4f} | "
            f"{group['evidence_recall_mean']:.4f} | "
            f"{group['valid_path_rate_mean']:.4f} | "
            f"{group['mean_rounds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Direct recurrent-state interventions",
            "",
            "| Intervention | Mean degradation | Seeds ≥ 0.05 | "
            "Causal forward ablation |",
            "|---|---:|---:|---|",
        ]
    )
    for name, item in summary["state_use"]["by_intervention"].items():
        lines.append(
            f"| {name} | {item['mean_degradation']:+.4f} | "
            f"{item['seeds_at_or_above_0_05']}/3 | "
            f"{item['causal_forward_ablation']} |"
        )
    lines.extend(
        [
            "",
            f"Recurrent-advantage rule passed: "
            f"**{summary['decision']['recurrent_advantage']}**.",
            "",
            f"Material-state-use rule passed: "
            f"**{summary['decision']['material_state_use']}**.",
            "",
            "Detach is expected to preserve forward evaluation values; it "
            "tests cross-round gradient flow only during training. Reset and "
            "graph-local shuffling are the registered causal forward tests.",
            "",
            f"All {summary['run_count']} runs report zero sealed access, zero "
            "deterministic replay mismatches, and zero row-permutation "
            "decision mismatches.",
            "",
        ]
    )
    return "\n".join(lines)


def load_enriched_records(
    run_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Path],
    dict[str, dict[str, Any]],
]:
    """Load compact worker records enriched with evaluator report fields."""

    run_directories = {
        path.parent.name: path.parent
        for path in run_root.glob("*/*/experiment_record.json")
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
        verification[experiment_id] = _verify_run(run_directory)
        record = _read_json(run_directory / "experiment_record.json")
        metrics = _read_json(run_directory / "run" / "metrics.json")
        enriched = {
            **record,
            "dataset_version": metrics.get("dataset_version"),
            "dataset_hashes": metrics.get("dataset_hashes"),
            "reports": metrics.get("reports"),
        }
        validate_record(enriched)
        records.append(enriched)
    return records, run_directories, verification


def validate_drive_backup(
    backup: Mapping[str, Any],
    run_directories: Mapping[str, Path],
    verification: Mapping[str, Mapping[str, Any]],
) -> None:
    """Cross-check every Drive ID against its local certified artifact."""

    folder = _mapping(backup.get("folder"), field="drive.folder")
    folder_id = folder.get("id")
    if not isinstance(folder_id, str) or not folder_id:
        raise RuntimeError("Drive backup folder ID is missing")
    runs = _mapping(backup.get("runs"), field="drive.runs")
    if set(runs) != set(EXPECTED_RUN_IDS):
        raise RuntimeError("Drive backup does not contain the frozen run set")
    drive_ids: set[str] = set()
    expected_checkpoint_names = {
        "checkpoint.pt",
        *{
            f"checkpoint_step_{step:06d}.pt"
            for step in (1_000, 2_000, 3_000, 4_000, 5_000)
        },
    }
    for experiment_id in EXPECTED_RUN_IDS:
        artifacts = _mapping(
            runs[experiment_id],
            field=f"drive.runs.{experiment_id}",
        )
        expected_names = expected_checkpoint_names | {
            f"{experiment_id}-result.zip"
        }
        if set(artifacts) != expected_names:
            raise RuntimeError(
                f"{experiment_id}: Drive artifact set is incomplete"
            )
        checkpoints = _mapping(
            verification[experiment_id]["checkpoints"],
            field=f"verification.{experiment_id}.checkpoints",
        )
        for name in expected_checkpoint_names:
            remote = _mapping(
                artifacts[name],
                field=f"drive.{experiment_id}.{name}",
            )
            local = _mapping(
                checkpoints[name],
                field=f"verification.{experiment_id}.{name}",
            )
            _validate_drive_entry(
                remote,
                expected_bytes=int(local["bytes"]),
                expected_sha256=str(local["sha256"]),
                folder_id=folder_id,
                drive_ids=drive_ids,
                field=f"{experiment_id}.{name}",
            )
        archive_name = f"{experiment_id}-result.zip"
        archive_path = (
            run_directories[experiment_id].parent / archive_name
        )
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        _validate_drive_entry(
            _mapping(
                artifacts[archive_name],
                field=f"drive.{experiment_id}.{archive_name}",
            ),
            expected_bytes=archive_path.stat().st_size,
            expected_sha256=sha256(archive_path),
            folder_id=folder_id,
            drive_ids=drive_ids,
            field=f"{experiment_id}.{archive_name}",
        )


def _validate_drive_entry(
    entry: Mapping[str, Any],
    *,
    expected_bytes: int,
    expected_sha256: str,
    folder_id: str,
    drive_ids: set[str],
    field: str,
) -> None:
    if entry.get("bytes") != expected_bytes:
        raise RuntimeError(f"{field}: Drive byte count mismatch")
    if entry.get("sha256") != expected_sha256:
        raise RuntimeError(f"{field}: Drive SHA-256 mismatch")
    if entry.get("drive_parent_verified") is not True:
        raise RuntimeError(f"{field}: Drive parent was not verified")
    if entry.get("drive_parent_id") != folder_id:
        raise RuntimeError(f"{field}: Drive parent folder mismatch")
    if entry.get("drive_size_verified") is not True:
        raise RuntimeError(f"{field}: Drive size was not verified")
    drive_id = entry.get("drive_id")
    if not isinstance(drive_id, str) or not drive_id:
        raise RuntimeError(f"{field}: Drive file ID is missing")
    if drive_id in drive_ids:
        raise RuntimeError(f"{field}: duplicate Drive file ID")
    drive_ids.add(drive_id)
    url = entry.get("drive_url")
    if not isinstance(url, str) or drive_id not in url:
        raise RuntimeError(f"{field}: Drive URL/ID mismatch")


def write_outputs(
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "training_experiments.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in records
        )
    )
    (output_root / "TRAINING_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "TRAINING_SUMMARY.md").write_text(
        render_markdown(summary)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/spider_v0_2/training/isolated"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/spider_v0_2/training"),
    )
    parser.add_argument(
        "--drive-backup",
        type=Path,
        default=Path("artifacts/spider_v0_2/GOOGLE_DRIVE_BACKUP.json"),
    )
    args = parser.parse_args()
    records, run_directories, verification = load_enriched_records(
        args.run_root
    )
    drive_backup = _read_json(args.drive_backup)
    validate_drive_backup(
        drive_backup,
        run_directories,
        verification,
    )
    summary = build_summary(records)
    summary["drive_backup"] = {
        "folder_id": drive_backup["folder"]["id"],
        "folder_url": drive_backup["folder"]["url"],
        "verified_artifact_count": 42,
    }
    write_outputs(args.output_root, records, summary)
    print(
        json.dumps(
            {
                "material_state_use": summary["decision"][
                    "material_state_use"
                ],
                "mean_structural_delta": summary["paired"][
                    "mean_structural_delta"
                ],
                "recurrent_advantage": summary["decision"][
                    "recurrent_advantage"
                ],
                "run_count": summary["run_count"],
                "sealed_access_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
