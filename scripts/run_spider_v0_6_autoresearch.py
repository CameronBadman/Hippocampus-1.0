#!/usr/bin/env python3
"""Run and summarize calibration-free Spider v0.6 experiments."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
from typing import Any
import zlib

from style_presets import rcparams


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_6/local_rtx5070ti"
SEEDS = (1701, 1802, 1903)
ARMS = ("Z0", "Z1", "Z2")
CONFIGS = {arm: ROOT / f"configs/spider_v0_6/{arm}.json" for arm in ARMS}
TRAIN_STEPS = 2_000
TARGET_SCORE = 0.82
MIN_DELTA = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Spider v0.6 zero-shot evidence campaign."
    )
    parser.add_argument(
        "--phase",
        choices=("run", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=int, default=295)
    parser.add_argument("--max-attempts", type=int, default=16)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl_once(
    path: Path,
    record: dict[str, Any],
    *,
    identity: str = "experiment_id",
) -> None:
    existing = (
        {
            json.loads(line)[identity]
            for line in path.read_text().splitlines()
            if line.strip()
        }
        if path.is_file()
        else set()
    )
    if record[identity] in existing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@contextmanager
def _campaign_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".campaign.lock"
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another v0.6 orchestrator holds {lock_path}"
            ) from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _base_command(
    *,
    arm: str,
    seed: int,
    experiment_id: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/spider_v0_4_phase_b.py"),
        "--config",
        str(CONFIGS[arm]),
        "--experiment-id",
        experiment_id,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--train-cases",
        "8192",
        "--selection-cases",
        "512",
        "--calibration-cases",
        "512",
        "--evaluation-cases",
        "1024",
        "--stop-after-steps",
        str(TRAIN_STEPS),
    ]


def _interrupted_stage(output_dir: Path) -> tuple[str, Path | None]:
    if (output_dir / "evaluation_pause.json").is_file():
        return "evaluation", None
    if (output_dir / "checkpoint.pt").is_file():
        return "selection", None
    partial = sorted(output_dir.glob("checkpoint_step_*.pt"))
    if partial:
        return "training", partial[-1]
    raise RuntimeError(f"incomplete run has no resumable checkpoint: {output_dir}")


def _attempt_command(
    *,
    base: list[str],
    output_dir: Path,
    source_commit: str,
) -> tuple[list[str], str]:
    if not output_dir.exists():
        return base + ["--pause-after-selection"], "training"
    stage, checkpoint = _interrupted_stage(output_dir)
    if stage == "evaluation":
        pause = _load(output_dir / "evaluation_pause.json")
        return (
            base
            + [
                "--resume-evaluation",
                "--training-source-commit",
                str(pause["training_source_commit"]),
            ],
            stage,
        )
    if stage == "selection":
        return (
            base
            + [
                "--resume-selection",
                "--pause-after-selection",
                "--training-source-commit",
                source_commit,
            ],
            stage,
        )
    assert checkpoint is not None
    return (
        base
        + [
            "--resume-training",
            "--resume-checkpoint",
            str(checkpoint),
            "--pause-after-selection",
            "--training-source-commit",
            source_commit,
        ],
        stage,
    )


def _validate_zero_shot(metrics: dict[str, Any], *, arm: str) -> None:
    if metrics["config_sha256"] != _sha256(CONFIGS[arm]):
        raise RuntimeError(f"{metrics['experiment_id']} config hash drifted")
    if metrics["sealed_access_count"] != 0:
        raise RuntimeError(f"{metrics['experiment_id']} records sealed access")
    if metrics["evidence_operating_policy"] not in {
        "learned_null",
        "candidate_null",
    }:
        raise RuntimeError("v0.6 result used a calibrated evidence policy")
    temperature = metrics["calibration"]["calibration"]["temperature"]
    if temperature["accepted"] or temperature["applied_temperature"] != 1.0:
        raise RuntimeError("v0.6 result fitted a held-out temperature")
    guards = metrics["guards"]
    if not guards["finite"]:
        raise RuntimeError("v0.6 result contains a non-finite metric")
    if guards["deterministic_replay_mismatches"] != 0:
        raise RuntimeError("v0.6 deterministic replay guard failed")
    if guards["row_permutation_decision_mismatches"] != 0:
        raise RuntimeError("v0.6 row-permutation guard failed")


def _run_or_load(
    *,
    arm: str,
    seed: int,
    output_root: Path,
    timeout_seconds: int,
    max_attempts: int,
) -> dict[str, Any]:
    experiment_id = f"V06-{arm}-s{seed}"
    output_dir = output_root / "runs" / experiment_id
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = _load(metrics_path)
        _validate_zero_shot(metrics, arm=arm)
        return metrics

    source_commit = _source_commit()
    base = _base_command(
        arm=arm,
        seed=seed,
        experiment_id=experiment_id,
        output_dir=output_dir,
    )
    log_path = output_root / "logs" / f"{experiment_id}.log"
    for attempt in range(1, max_attempts + 1):
        command, stage = _attempt_command(
            base=base,
            output_dir=output_dir,
            source_commit=source_commit,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("a" if output_dir.exists() else "w") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
        except subprocess.TimeoutExpired as error:
            _append_jsonl_once(
                output_root / "attempts.jsonl",
                {
                    "attempt_id": f"{experiment_id}-{attempt}",
                    "experiment_id": experiment_id,
                    "timestamp": _now(),
                    "stage": stage,
                    "status": "timeout_resumable",
                    "timeout_seconds": timeout_seconds,
                    "failure_reason": str(error),
                    "sealed_access_count": 0,
                },
                identity="attempt_id",
            )
            continue
        if completed.returncode != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-20:])
            raise RuntimeError(
                f"{experiment_id} failed in {stage} with "
                f"exit {completed.returncode}:\n{tail}"
            )
        if metrics_path.is_file():
            metrics = _load(metrics_path)
            _validate_zero_shot(metrics, arm=arm)
            return metrics
    raise TimeoutError(
        f"{experiment_id} did not finish after {max_attempts} bounded attempts"
    )


def _metric(row: dict[str, Any], name: str) -> float:
    return float(row["primary_metric"][name])


def _score(row: dict[str, Any]) -> float:
    if _metric(row, "scored_positive_coverage") < 0.98:
        return 0.0
    return min(
        _metric(row, "exact_evidence_set_accuracy"),
        _metric(row, "precision"),
        _metric(row, "recall"),
    )


def _ledger_record(metrics: dict[str, Any], *, arm: str) -> dict[str, Any]:
    return {
        "experiment_id": metrics["experiment_id"],
        "timestamp": _now(),
        "source_commit": metrics["source_commit"],
        "dataset_hash": metrics["dataset_hash"],
        "config_sha256": _sha256(CONFIGS[arm]),
        "arm": arm,
        "seed": metrics["resolved_seed"],
        "status": metrics["status"],
        "score": _score(metrics),
        "parameter_count": metrics["parameter_count"],
        "planned_steps": metrics["planned_steps"],
        "completed_steps": metrics["completed_steps"],
        "selected_step": metrics["selected_step"],
        "primary_metric": metrics["primary_metric"],
        "per_family": metrics["per_family"],
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
        "selected_checkpoint_sha256": metrics[
            "selected_checkpoint_sha256"
        ],
        "temperature_fitted": False,
        "sealed_access_count": 0,
    }


def _arm_summary(
    results: dict[tuple[str, int], dict[str, Any]],
    arm: str,
) -> dict[str, float]:
    fields = (
        "exact_evidence_set_accuracy",
        "precision",
        "recall",
        "scored_positive_coverage",
        "macro_average_precision",
        "false_positives_per_case",
        "mean_worst_positive_rank",
        "mean_absolute_cardinality_error",
    )
    rows = [results[(arm, seed)] for seed in SEEDS]
    summary: dict[str, float] = {}
    for field in fields:
        values = [
            _metric(row, field)
            for row in rows
            if row["primary_metric"][field] is not None
        ]
        summary[field] = statistics.mean(values) if values else math.inf
    summary["mean_selected_step"] = statistics.mean(
        float(row["selected_step"]) for row in rows
    )
    summary["score"] = min(
        summary["exact_evidence_set_accuracy"],
        summary["precision"],
        summary["recall"],
    )
    return summary


def _family_summary(
    results: dict[tuple[str, int], dict[str, Any]],
    arm: str,
) -> dict[str, dict[str, float]]:
    fields = (
        "exact_evidence_set_accuracy",
        "precision",
        "recall",
        "macro_evidence_average_precision",
        "false_positives_per_case",
    )
    families = tuple(sorted(results[(arm, SEEDS[0])]["per_family"]))
    return {
        family: {
            field: statistics.mean(
                float(results[(arm, seed)]["per_family"][family][field])
                for seed in SEEDS
            )
            for field in fields
        }
        for family in families
    }


def _load_all(
    output_root: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = output_root / "runs" / f"V06-{arm}-s{seed}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            metrics = _load(path)
            _validate_zero_shot(metrics, arm=arm)
            results[(arm, seed)] = metrics
    return results


def _set_pixel(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = 3 * (y * width + x)
        image[offset : offset + 3] = bytes(color)


def _line(
    image: bytearray,
    width: int,
    height: int,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = first
    x1, y1 = second
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _set_pixel(image, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_progress(output_root: Path, records: list[dict[str, Any]]) -> None:
    style = rcparams()
    width = int(style["width"])
    height = int(style["height"])
    margin = int(style["margin"])
    image = bytearray(bytes(style["background"]) * (width * height))
    for fraction in (0.25, 0.5, 0.75, 1.0):
        y = round(height - margin - fraction * (height - 2 * margin))
        _line(
            image,
            width,
            height,
            (margin, y),
            (width - margin, y),
            style["grid"],
        )
    points: list[tuple[int, int]] = []
    best_points: list[tuple[int, int]] = []
    best = 0.0
    denominator = max(1, len(records) - 1)
    for index, record in enumerate(records):
        score = float(record["score"])
        x = margin + round(index / denominator * (width - 2 * margin))
        y = height - margin - round(
            max(0.0, min(1.0, score)) * (height - 2 * margin)
        )
        points.append((x, y))
        best = max(best, score)
        best_points.append(
            (x, height - margin - round(best * (height - 2 * margin)))
        )
        for offset_x in range(-4, 5):
            for offset_y in range(-4, 5):
                if offset_x * offset_x + offset_y * offset_y <= 16:
                    _set_pixel(
                        image,
                        width,
                        height,
                        x + offset_x,
                        y + offset_y,
                        style["kept"],
                    )
    for first, second in zip(points, points[1:], strict=False):
        _line(image, width, height, first, second, style["kept"])
    for first, second in zip(best_points, best_points[1:], strict=False):
        _line(image, width, height, first, second, style["best"])
    rows = b"".join(
        b"\x00" + bytes(image[y * width * 3 : (y + 1) * width * 3])
        for y in range(height)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(rows, level=9))
        + _png_chunk(b"IEND", b"")
    )
    (output_root / "progress.png").write_bytes(png)


def _summarize(output_root: Path) -> dict[str, Any]:
    results = _load_all(output_root)
    summaries = {arm: _arm_summary(results, arm) for arm in ARMS}
    per_family = {arm: _family_summary(results, arm) for arm in ARMS}
    candidate_gates: dict[str, dict[str, Any]] = {}
    for arm in ARMS[1:]:
        seed_results = []
        for seed in SEEDS:
            control = _score(results[("Z0", seed)])
            candidate = _score(results[(arm, seed)])
            seed_results.append(
                {
                    "seed": seed,
                    "control_score": control,
                    "candidate_score": candidate,
                    "delta": candidate - control,
                    "advances": candidate >= control + MIN_DELTA - 1e-12,
                }
            )
        wins = sum(int(row["advances"]) for row in seed_results)
        candidate_gates[arm] = {
            "seed_results": seed_results,
            "seed_wins": wins,
            "candidate_success": (
                wins >= 2
                and summaries[arm]["score"] >= TARGET_SCORE
                and summaries[arm]["scored_positive_coverage"] >= 0.98
            ),
        }
    successful = [
        arm
        for arm in ARMS[1:]
        if candidate_gates[arm]["candidate_success"]
    ]
    finalist = (
        max(successful, key=lambda arm: summaries[arm]["score"])
        if successful
        else "Z0"
    )
    payload = {
        "campaign": "Spider v0.6 zero-shot evidence energy",
        "source_commit": _source_commit(),
        "dataset_hash": next(iter(results.values()))["dataset_hash"],
        "target_score": TARGET_SCORE,
        "score_definition": "min(exact_set, precision, recall)",
        "arm_summaries": summaries,
        "per_family": per_family,
        "gate": {"candidates": candidate_gates},
        "selected_finalist": finalist,
        "accepted_training_run_count": len(ARMS) * len(SEEDS),
        "temperature_fit_count": 0,
        "symbol_overlap_count": 0,
        "sealed_access_count": 0,
        "run_source_commits": sorted(
            {row["source_commit"] for row in results.values()}
        ),
    }
    _write(output_root / "SUMMARY.json", payload)
    lines = [
        "# Spider v0.6 zero-shot evidence results",
        "",
        "| Arm | Score | Exact set | Precision | Recall | Coverage | Macro AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = summaries[arm]
        lines.append(
            f"| {arm} | {row['score']:.4f} | "
            f"{row['exact_evidence_set_accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['scored_positive_coverage']:.4f} | "
            f"{row['macro_average_precision']:.4f} |"
        )
    lines.extend(
        (
            "",
            f"Selected finalist: `{finalist}`.",
            "",
            "No temperature, threshold, or cardinality policy was fitted.",
            "No sealed split was materialised or evaluated.",
        )
    )
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    records = [
        _ledger_record(results[(arm, seed)], arm=arm)
        for arm in ARMS
        for seed in SEEDS
    ]
    _write_progress(output_root, records)
    _write(
        output_root / "FINALIST.json",
        {
            "selected_arm": finalist,
            "selection_reason": (
                f"{finalist} passed the zero-shot gate"
                if finalist != "Z0"
                else "no candidate passed the zero-shot gate"
            ),
            "dataset_hash": payload["dataset_hash"],
            "score": summaries[finalist]["score"],
            "checkpoints": [
                {
                    "seed": seed,
                    "selected_step": results[(finalist, seed)][
                        "selected_step"
                    ],
                    "checkpoint_sha256": results[(finalist, seed)][
                        "selected_checkpoint_sha256"
                    ],
                    "source_commit": results[(finalist, seed)][
                        "source_commit"
                    ],
                }
                for seed in SEEDS
            ],
            "temperature_fit_count": 0,
            "symbol_overlap_count": 0,
            "sealed_access_count": 0,
        },
    )
    return payload


def _run_campaign(args: argparse.Namespace) -> None:
    if args.phase in {"run", "all"}:
        arms = (args.arm,) if args.arm else ARMS
        seeds = (args.seed,) if args.seed else SEEDS
        for arm in arms:
            for seed in seeds:
                metrics = _run_or_load(
                    arm=arm,
                    seed=seed,
                    output_root=args.output_root,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=args.max_attempts,
                )
                _append_jsonl_once(
                    args.output_root / "experiments.jsonl",
                    _ledger_record(metrics, arm=arm),
                )
        if args.arm is not None or args.seed is not None:
            print(
                json.dumps(
                    {
                        "completed_arms": list(arms),
                        "completed_seeds": list(seeds),
                        "summary_pending": True,
                    },
                    sort_keys=True,
                )
            )
            return
    result = _summarize(args.output_root)
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0 or args.timeout_seconds > 300:
        raise ValueError("timeout-seconds must be in [1, 300]")
    if args.max_attempts <= 0:
        raise ValueError("max-attempts must be positive")
    with _campaign_lock(args.output_root):
        _run_campaign(args)


if __name__ == "__main__":
    main()
