#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_4/phase_b/local_rtx5070ti"
SEEDS = (1701, 1802, 1903)
ARMS = ("B0", "B1", "B2")
CONFIGS = {
    arm: ROOT / f"configs/spider_v0_4/phase_b_{arm}.json"
    for arm in ARMS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the gated Spider v0.4 AutoResearch campaign."
    )
    parser.add_argument(
        "--phase",
        choices=("B", "summarize"),
        default="B",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=int, default=300)
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _append_once(path: Path, record: dict[str, Any]) -> None:
    existing = (
        {
            json.loads(line)["experiment_id"]
            for line in path.read_text().splitlines()
            if line.strip()
        }
        if path.exists()
        else set()
    )
    if record["experiment_id"] in existing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _ledger_record(
    metrics: dict[str, Any],
    *,
    arm: str,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": metrics["experiment_id"],
        "timestamp": metrics["timestamp"],
        "source_commit": metrics["source_commit"],
        "dataset_hash": metrics["dataset_hash"],
        "config_sha256": _sha256(CONFIGS[arm]),
        "arm": arm,
        "seed": metrics["resolved_seed"],
        "renderer_geometry": metrics["renderer"]["geometry"],
        "parameter_count": metrics["parameter_count"],
        "planned_steps": metrics["planned_steps"],
        "selected_step": metrics["selected_step"],
        "status": metrics["status"],
        "pass": metrics["pass"],
        "sealed_access_count": metrics["sealed_access_count"],
        "primary_metric": metrics["primary_metric"],
        "per_family": metrics["per_family"],
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
        "selected_checkpoint_sha256": metrics[
            "selected_checkpoint_sha256"
        ],
        "output_dir": str(output_dir),
    }


def _run_or_load(
    *,
    arm: str,
    seed: int,
    output_root: Path,
    source_commit: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    experiment_id = f"V04-phase-B-{arm}-s{seed}"
    output_dir = output_root / "runs" / experiment_id
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _load(metrics_path)
        if metrics["source_commit"] != source_commit:
            raise RuntimeError(
                f"{experiment_id} was produced by another source commit"
            )
        return metrics
    if output_dir.exists():
        raise RuntimeError(
            f"incomplete run requires manual preservation: {output_dir}"
        )
    log_path = output_root / "logs" / f"{experiment_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
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
    ]
    with log_path.open("w") as log:
        try:
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
            _append_once(
                output_root / "failed_experiments.jsonl",
                {
                    "experiment_id": experiment_id,
                    "timestamp": _now(),
                    "source_commit": source_commit,
                    "arm": arm,
                    "seed": seed,
                    "status": "timeout",
                    "timeout_seconds": timeout_seconds,
                    "failure_reason": str(error),
                    "sealed_access_count": 0,
                },
            )
            raise
    if completed.returncode != 0 or not metrics_path.exists():
        _append_once(
            output_root / "failed_experiments.jsonl",
            {
                "experiment_id": experiment_id,
                "timestamp": _now(),
                "source_commit": source_commit,
                "arm": arm,
                "seed": seed,
                "status": "crashed",
                "returncode": completed.returncode,
                "log_path": str(log_path),
                "sealed_access_count": 0,
            },
        )
        raise RuntimeError(f"{experiment_id} failed; inspect {log_path}")
    return _load(metrics_path)


def _renderer_decision(
    results: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, object]:
    seed_rows: list[dict[str, object]] = []
    wins = 0
    for seed in SEEDS:
        control = results[("B0", seed)]
        aligned = results[("B2", seed)]
        control_lookup = float(control["per_family"]["lookup"]["recall"])
        aligned_lookup = float(aligned["per_family"]["lookup"]["recall"])
        control_reach = float(
            control["per_family"]["reachability"]["recall"]
        )
        aligned_reach = float(
            aligned["per_family"]["reachability"]["recall"]
        )
        control_exact = float(
            control["primary_metric"]["exact_evidence_set_accuracy"]
        )
        aligned_exact = float(
            aligned["primary_metric"]["exact_evidence_set_accuracy"]
        )
        control_precision = float(control["primary_metric"]["precision"])
        aligned_precision = float(aligned["primary_metric"]["precision"])
        deltas = {
            "lookup_recall": aligned_lookup - control_lookup,
            "reachability_recall": aligned_reach - control_reach,
            "exact_set": aligned_exact - control_exact,
            "precision": aligned_precision - control_precision,
        }
        won = (
            aligned["pass"]
            and control["pass"]
            and deltas["lookup_recall"] >= 0.30
            and deltas["reachability_recall"] >= 0.20
            and deltas["exact_set"] >= 0.10
            and deltas["precision"] >= -0.02
        )
        wins += int(won)
        seed_rows.append({"seed": seed, "deltas": deltas, "advance": won})
    return {
        "seed_results": seed_rows,
        "seed_wins": wins,
        "B2_advances": wins >= 2,
        "next_phase": "C" if wins >= 2 else "inspect_generator_and_labels",
    }


def _arm_summary(
    results: dict[tuple[str, int], dict[str, Any]],
    arm: str,
) -> dict[str, float]:
    rows = [results[(arm, seed)]["primary_metric"] for seed in SEEDS]
    fields = (
        "exact_evidence_set_accuracy",
        "precision",
        "recall",
        "scored_positive_coverage",
        "macro_average_precision",
        "false_positives_per_case",
    )
    return {
        field: statistics.mean(float(row[field]) for row in rows)
        for field in fields
    }


def _write_summary(
    output_root: Path,
    results: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, object]:
    decision = _renderer_decision(results)
    payload = {
        "campaign": "Spider v0.4 Phase B renderer causal test",
        "source_commit": _source_commit(),
        "dataset_hash": next(iter(results.values()))["dataset_hash"],
        "sealed_access_count": 0,
        "arms": {arm: _arm_summary(results, arm) for arm in ARMS},
        "decision": decision,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    checkpoint_index = {
        "source_commit": payload["source_commit"],
        "dataset_hash": payload["dataset_hash"],
        "sealed_access_count": 0,
        "checkpoints": [
            {
                "arm": arm,
                "seed": seed,
                "selected_step": results[(arm, seed)]["selected_step"],
                "checkpoint_sha256": results[(arm, seed)][
                    "selected_checkpoint_sha256"
                ],
                "historical_reuse": arm == "B0",
            }
            for arm in ARMS
            for seed in SEEDS
        ],
    }
    (output_root / "CHECKPOINT_INDEX.json").write_text(
        json.dumps(checkpoint_index, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Spider v0.4 Phase B",
        "",
        "| Arm | Exact set | Precision | Recall | Coverage | Macro AP | FP/case |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = payload["arms"][arm]
        lines.append(
            f"| {arm} | {row['exact_evidence_set_accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['scored_positive_coverage']:.4f} | "
            f"{row['macro_average_precision']:.4f} | "
            f"{row['false_positives_per_case']:.4f} |"
        )
    lines.extend(
        (
            "",
            f"B2 seed wins: {decision['seed_wins']}/3.",
            f"Advance: {decision['B2_advances']}.",
            f"Next phase: `{decision['next_phase']}`.",
            "",
            "No sealed split was accessed.",
        )
    )
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    args = parse_args()
    source_commit = _source_commit()
    results: dict[tuple[str, int], dict[str, Any]] = {}
    if args.phase == "summarize":
        for arm in ARMS:
            for seed in SEEDS:
                path = (
                    args.output_root
                    / "runs"
                    / f"V04-phase-B-{arm}-s{seed}"
                    / "metrics.json"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                results[(arm, seed)] = _load(path)
    else:
        selected_arms = (args.arm,) if args.arm is not None else ARMS
        selected_seeds = (args.seed,) if args.seed is not None else SEEDS
        for arm in selected_arms:
            for seed in selected_seeds:
                metrics = _run_or_load(
                    arm=arm,
                    seed=seed,
                    output_root=args.output_root,
                    source_commit=source_commit,
                    timeout_seconds=args.timeout_seconds,
                )
                results[(arm, seed)] = metrics
                _append_once(
                    args.output_root / "experiments.jsonl",
                    _ledger_record(
                        metrics,
                        arm=arm,
                        output_dir=(
                            args.output_root
                            / "runs"
                            / metrics["experiment_id"]
                        ),
                    ),
                )
    if len(results) == len(ARMS) * len(SEEDS):
        output = _write_summary(args.output_root, results)
    else:
        output = {
            "completed_experiments": sorted(
                metrics["experiment_id"] for metrics in results.values()
            ),
            "summary_pending": True,
        }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
