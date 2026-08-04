#!/usr/bin/env python3
"""Run and ledger the registered Spider v0.8 transfer experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_8/local_rtx5070ti"
CONFIGS = {
    arm: ROOT / f"configs/spider_v0_8/{arm}.json"
    for arm in ("T0", "T1", "T2")
}
SCREEN = (("T0", 1701), ("T1", 1701), ("T2", 1701))
CONFIRM = (
    ("T1", 1802),
    ("T1", 1903),
    ("T2", 1802),
    ("T2", 1903),
)
TARGET_SCORE = 0.85
COMPONENT_FLOOR = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("screen", "confirm", "all", "summarize"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=int, default=295)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _experiment_id(arm: str, seed: int) -> str:
    stage = "S" if seed == 1701 else "C"
    return f"V08{stage}-{arm}-s{seed}"


def _run_one(
    *,
    arm: str,
    seed: int,
    output_root: Path,
    timeout: int,
    device: str,
) -> dict[str, Any]:
    experiment_id = _experiment_id(arm, seed)
    run_dir = output_root / "runs" / experiment_id
    result_path = run_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    command = [
        sys.executable,
        str(ROOT / "scripts/train_spider_v0_8_sre.py"),
        "--config",
        str(CONFIGS[arm]),
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
        "--device",
        device,
    ]
    print(json.dumps({"event": "start", "experiment_id": experiment_id}), flush=True)
    started = _now()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        (output_root / "logs").mkdir(parents=True, exist_ok=True)
        (output_root / "logs" / f"{experiment_id}.log").write_text(
            (error.stdout or "") + "\n" + (error.stderr or "")
        )
        return {
            "experiment_id": experiment_id,
            "arm": arm,
            "seed": seed,
            "status": "invalid",
            "failure_reason": f"timeout after {timeout} seconds",
            "timestamp": started,
        }
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "logs" / f"{experiment_id}.log").write_text(
        completed.stdout + "\n" + completed.stderr
    )
    if completed.returncode or not result_path.is_file():
        return {
            "experiment_id": experiment_id,
            "arm": arm,
            "seed": seed,
            "status": "invalid",
            "failure_reason": f"runner exited {completed.returncode}",
            "timestamp": started,
        }
    result = json.loads(result_path.read_text())
    print(
        json.dumps(
            {
                "event": "complete",
                "experiment_id": experiment_id,
                "score": result["evaluation"]["score"],
            }
        ),
        flush=True,
    )
    return result


def _record(result: dict[str, Any]) -> dict[str, Any]:
    if "evaluation" not in result:
        return result
    arm = str(result["arm"])
    seed = int(result["seed"])
    metrics = result["evaluation"]
    components = metrics["components"]
    replay = result["deterministic_replay"]
    permutation = result["row_permutation"]
    finite = all(
        math.isfinite(float(value))
        for value in (metrics["score"], *components.values())
    )
    accepted = (
        finite
        and result["sealed_access_count"] == 0
        and replay["score_mismatch_count"] == 0
        and replay["null_mismatch_count"] == 0
        and replay["decision_mismatch_count"] == 0
        and permutation["decision_mismatch_count"] == 0
        and metrics["scored_positive_coverage"] == 1.0
        and metrics["enumerated_candidate_coverage"] == 1.0
    )
    return {
        "experiment_id": _experiment_id(arm, seed),
        "timestamp": result["timestamp"],
        "source_commit": result["source_commit"],
        "config_sha256": _sha256(CONFIGS[arm]),
        "arm": arm,
        "seed": seed,
        "status": "accepted" if accepted else "invalid",
        "parameter_count": result["parameter_count"],
        "selected_step": result["selected_step"],
        "score": metrics["score"],
        "components": components,
        "set_selection": metrics["set_selection"],
        "oracle_cardinality_top_rank_exact": metrics[
            "oracle_cardinality_top_rank_exact"
        ],
        "mean_worst_positive_rank": metrics["mean_worst_positive_rank"],
        "mean_positive_best_negative_margin": metrics[
            "mean_positive_best_negative_margin"
        ],
        "runtime_seconds": result["runtime_seconds"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
        "deterministic_replay": replay,
        "row_permutation": permutation,
        "sealed_access_count": result["sealed_access_count"],
        "failure_reason": None if accepted else "mechanical guard failed",
    }


def _collect(output_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((output_root / "runs").glob("V08*/result.json")):
        records.append(_record(json.loads(path.read_text())))
    return records


def _summarize(output_root: Path) -> None:
    records = _collect(output_root)
    ledger = output_root / "experiments.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    accepted = [record for record in records if record["status"] == "accepted"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in accepted:
        groups.setdefault(record["arm"], []).append(record)
    aggregate = {}
    for arm, values in sorted(groups.items()):
        aggregate[arm] = {
            "run_count": len(values),
            "mean_score": statistics.mean(value["score"] for value in values),
            "scores": [value["score"] for value in values],
            "mean_components": {
                name: statistics.mean(value["components"][name] for value in values)
                for name in values[0]["components"]
            },
        }
    passing = [
        (arm, values)
        for arm, values in aggregate.items()
        if values["mean_score"] >= TARGET_SCORE
        and min(values["mean_components"].values()) >= COMPONENT_FLOOR
    ]
    recommendation = (
        max(passing, key=lambda item: item[1]["mean_score"])[0]
        if passing
        else (
            max(aggregate, key=lambda arm: aggregate[arm]["mean_score"])
            if aggregate
            else None
        )
    )
    report = {
        "format": "spider-v0.8-sre-aggregate-v1",
        "generated_at": _now(),
        "source_commit": _source_commit(),
        "target_score": TARGET_SCORE,
        "component_floor": COMPONENT_FLOOR,
        "accepted_run_count": len(accepted),
        "invalid_run_count": len(records) - len(accepted),
        "sealed_access_count": 0,
        "arms": aggregate,
        "recommendation": recommendation,
        "target_met": bool(passing),
    }
    _write(output_root / "aggregate_report.json", report)
    lines = [
        "# Spider v0.8 SRE experiment ledger",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "| Arm | Accepted runs | Mean score | MRR | Recall@8 | Macro AP | Pairwise |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, values in aggregate.items():
        component = values["mean_components"]
        lines.append(
            f"| {arm} | {values['run_count']} | {values['mean_score']:.4f} | "
            f"{component['mrr']:.4f} | {component['recall_at_8']:.4f} | "
            f"{component['macro_average_precision']:.4f} | "
            f"{component['macro_hard_negative_pairwise_accuracy']:.4f} |"
        )
    lines.extend(
        (
            "",
            f"Recommendation: **{recommendation or 'none'}**.",
            "",
            f"Registered target met: **{report['target_met']}**.",
            "",
            "The SRE sealed test was not accessed.",
        )
    )
    (output_root / "EXPERIMENTS.md").write_text("\n".join(lines) + "\n")
    research_log = ROOT / "research/spider_v0_8/log.jsonl"
    research_log.parent.mkdir(parents=True, exist_ok=True)
    research_log.write_text(
        "".join(
            json.dumps(
                {
                    "iteration": index + 1,
                    "hypothesis": {
                        "T0": "Frozen semantic/status control",
                        "T1": "Packed canonical scorer without alignment",
                        "T2": "Multi-positive canonical alignment",
                    }[record["arm"]],
                    "experiment_id": record["experiment_id"],
                    "status": record["status"],
                    "score": record.get("score"),
                    "failure_reason": record.get("failure_reason"),
                },
                sort_keys=True,
            )
            + "\n"
            for index, record in enumerate(records)
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.stage != "summarize":
        experiments = []
        if args.stage in {"screen", "all"}:
            experiments.extend(SCREEN)
        if args.stage in {"confirm", "all"}:
            experiments.extend(CONFIRM)
        for arm, seed in experiments:
            result = _run_one(
                arm=arm,
                seed=seed,
                output_root=args.output_root,
                timeout=args.timeout_seconds,
                device=args.device,
            )
            if "evaluation" not in result:
                print(json.dumps(result, sort_keys=True), flush=True)
    _summarize(args.output_root)


if __name__ == "__main__":
    main()
