#!/usr/bin/env python3
"""Run the registered Spider v0.7 canonical-binding campaign."""

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
import subprocess
import sys
from typing import Any

from run_spider_v0_6_autoresearch import _write_progress


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_7/local_rtx5070ti"
V06_OUTPUT = ROOT / "artifacts/spider_v0_6/local_rtx5070ti"
SEEDS = (1701, 1802, 1903)
SCREEN_ARMS = ("R0", "R1", "R2")
CONFIRM_ARMS = ("R0", "R2")
CONFIGS = {
    arm: ROOT / f"configs/spider_v0_7/{arm}.json"
    for arm in SCREEN_ARMS
}
TARGET_SCORE = 0.82
PRECISION_FLOOR = 0.90
COVERAGE_FLOOR = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Spider v0.7 canonical-binding campaign."
    )
    parser.add_argument(
        "--phase",
        choices=("run", "summarize", "all"),
        default="all",
    )
    parser.add_argument(
        "--stage",
        choices=("screen", "confirm"),
        default="screen",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=int, default=295)
    parser.add_argument("--max-attempts", type=int, default=16)
    parser.add_argument("--arm", choices=SCREEN_ARMS)
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


def _append_jsonl_once(path: Path, record: dict[str, Any]) -> None:
    existing = (
        {
            json.loads(line)["experiment_id"]
            for line in path.read_text().splitlines()
            if line.strip()
        }
        if path.is_file()
        else set()
    )
    if record["experiment_id"] in existing:
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
                f"another v0.7 orchestrator holds {lock_path}"
            ) from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _v06_z1_checkpoint(seed: int) -> Path:
    rows = [
        json.loads(line)
        for line in (V06_OUTPUT / "experiments.jsonl").read_text().splitlines()
        if line.strip()
    ]
    record = next(
        row for row in rows
        if row["arm"] == "Z1" and int(row["seed"]) == seed
    )
    step = int(record["selected_step"])
    run = V06_OUTPUT / "runs" / f"V06-Z1-s{seed}"
    checkpoint = (
        run / "checkpoint.pt"
        if step == int(record["completed_steps"])
        else run / f"checkpoint_step_{step:06d}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if _sha256(checkpoint) != record["selected_checkpoint_sha256"]:
        raise RuntimeError("historical Z1 checkpoint hash drifted")
    return checkpoint


def _stage_settings(stage: str) -> tuple[tuple[str, ...], int, str, str]:
    if stage == "screen":
        return SCREEN_ARMS, 1_000, "evidence", "V07S"
    return CONFIRM_ARMS, 2_000, "all", "V07F"


def _base_command(
    *,
    arm: str,
    seed: int,
    experiment_id: str,
    output_dir: Path,
    stage: str,
) -> list[str]:
    _, steps, scope, _ = _stage_settings(stage)
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
        str(steps),
        "--initial-checkpoint",
        str(_v06_z1_checkpoint(seed)),
        "--trainable-scope",
        scope,
    ]


def _interrupted_stage(output_dir: Path) -> tuple[str, Path | None]:
    if (output_dir / "evaluation_pause.json").is_file():
        return "evaluation", None
    if (output_dir / "checkpoint.pt").is_file():
        return "selection", None
    partial = sorted(output_dir.glob("checkpoint_step_*.pt"))
    if partial:
        return "training", partial[-1]
    raise RuntimeError(f"incomplete run has no checkpoint: {output_dir}")


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


def _validate(metrics: dict[str, Any], *, arm: str) -> None:
    if metrics["config_sha256"] != _sha256(CONFIGS[arm]):
        raise RuntimeError("v0.7 config hash drifted")
    if metrics["dataset_version"] != "spider-programs-v0.7-binding-dev":
        raise RuntimeError("v0.7 dataset version drifted")
    if metrics["sealed_access_count"] != 0:
        raise RuntimeError("v0.7 result records sealed access")
    if metrics["evidence_operating_policy"] != "candidate_null":
        raise RuntimeError("v0.7 result changed the zero-shot policy")
    temperature = metrics["calibration"]["calibration"]["temperature"]
    if temperature["accepted"] or temperature["applied_temperature"] != 1.0:
        raise RuntimeError("v0.7 result fitted a temperature")
    guards = metrics["guards"]
    if not guards["finite"]:
        raise RuntimeError("v0.7 result contains a non-finite metric")
    if guards["deterministic_replay_mismatches"] != 0:
        raise RuntimeError("v0.7 deterministic replay guard failed")
    if guards["row_permutation_decision_mismatches"] != 0:
        raise RuntimeError("v0.7 row-permutation guard failed")
    if not metrics["binding_retrieval"]["diagnostic_only"]:
        raise RuntimeError("v0.7 binding audit altered the model")


def _run_or_load(
    *,
    arm: str,
    seed: int,
    stage: str,
    output_root: Path,
    timeout_seconds: int,
    max_attempts: int,
) -> dict[str, Any]:
    _, _, _, prefix = _stage_settings(stage)
    experiment_id = f"{prefix}-{arm}-s{seed}"
    stage_root = output_root / stage
    output_dir = stage_root / "runs" / experiment_id
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = _load(metrics_path)
        _validate(metrics, arm=arm)
        return metrics
    source_commit = _source_commit()
    base = _base_command(
        arm=arm,
        seed=seed,
        experiment_id=experiment_id,
        output_dir=output_dir,
        stage=stage,
    )
    log_path = stage_root / "logs" / f"{experiment_id}.log"
    for attempt in range(1, max_attempts + 1):
        command, attempt_stage = _attempt_command(
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
                stage_root / "attempts.jsonl",
                {
                    "experiment_id": f"{experiment_id}-attempt-{attempt}",
                    "parent_experiment_id": experiment_id,
                    "timestamp": _now(),
                    "stage": attempt_stage,
                    "status": "timeout_resumable",
                    "failure_reason": str(error),
                    "sealed_access_count": 0,
                },
            )
            continue
        if completed.returncode != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
            raise RuntimeError(
                f"{experiment_id} failed in {attempt_stage}:\n{tail}"
            )
        if metrics_path.is_file():
            metrics = _load(metrics_path)
            _validate(metrics, arm=arm)
            return metrics
    raise TimeoutError(f"{experiment_id} exhausted its bounded attempts")


def _metric(row: dict[str, Any], name: str) -> float:
    return float(row["primary_metric"][name])


def _score(row: dict[str, Any]) -> float:
    if _metric(row, "scored_positive_coverage") < COVERAGE_FLOOR:
        return 0.0
    return min(
        _metric(row, "exact_evidence_set_accuracy"),
        _metric(row, "precision"),
        _metric(row, "recall"),
    )


def _ledger_record(metrics: dict[str, Any], *, arm: str, stage: str) -> dict[str, Any]:
    return {
        "experiment_id": metrics["experiment_id"],
        "timestamp": _now(),
        "source_commit": metrics["source_commit"],
        "dataset_hash": metrics["dataset_hash"],
        "config_sha256": _sha256(CONFIGS[arm]),
        "stage": stage,
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
        "binding_retrieval": metrics["binding_retrieval"],
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
        "selected_checkpoint_sha256": metrics["selected_checkpoint_sha256"],
        "temperature_fitted": False,
        "sealed_access_count": 0,
    }


def _load_all(output_root: Path, stage: str) -> dict[tuple[str, int], dict[str, Any]]:
    arms, _, _, prefix = _stage_settings(stage)
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in arms:
        for seed in SEEDS:
            path = (
                output_root / stage / "runs" / f"{prefix}-{arm}-s{seed}" / "metrics.json"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            metrics = _load(path)
            _validate(metrics, arm=arm)
            results[(arm, seed)] = metrics
    return results


def _arm_summary(results, arm: str) -> dict[str, float]:
    fields = (
        "exact_evidence_set_accuracy",
        "precision",
        "recall",
        "scored_positive_coverage",
        "macro_average_precision",
        "false_positives_per_case",
    )
    summary = {
        field: statistics.mean(
            _metric(results[(arm, seed)], field) for seed in SEEDS
        )
        for field in fields
    }
    summary["score"] = min(
        summary["exact_evidence_set_accuracy"],
        summary["precision"],
        summary["recall"],
    )
    return summary


def _family_summary(results, arm: str) -> dict[str, dict[str, float]]:
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


def _canonical_minimum(metrics: dict[str, Any], field: str) -> float:
    stages = metrics["binding_retrieval"]["stages"]
    canonical = next(
        (stage for stage in stages if stage["stage"] == "evidence_canonical"),
        None,
    )
    return 0.0 if canonical is None else float(canonical[field])


def _screen_gate(results, summaries, families) -> dict[str, Any]:
    seed_results = []
    for seed in SEEDS:
        r0 = results[("R0", seed)]
        r1 = results[("R1", seed)]
        r2 = results[("R2", seed)]
        lookup_gain = (
            float(r2["per_family"]["lookup"]["recall"])
            - float(r1["per_family"]["lookup"]["recall"])
        )
        exact_gain = (
            _metric(r2, "exact_evidence_set_accuracy")
            - _metric(r0, "exact_evidence_set_accuracy")
        )
        advances = lookup_gain >= 0.50 and exact_gain >= 0.05
        seed_results.append(
            {
                "seed": seed,
                "lookup_recall_gain_over_R1": lookup_gain,
                "exact_set_gain_over_R0": exact_gain,
                "advances": advances,
            }
        )
    latest_regression = (
        families["R0"]["latest_valid"]["recall"]
        - families["R2"]["latest_valid"]["recall"]
    )
    corroboration_regression = (
        families["R0"]["corroboration"]["recall"]
        - families["R2"]["corroboration"]["recall"]
    )
    retrieval_auroc = statistics.mean(
        _canonical_minimum(
            results[("R2", seed)],
            "minimum_direct_auroc",
        )
        for seed in SEEDS
    )
    retrieval_top1 = statistics.mean(
        _canonical_minimum(
            results[("R2", seed)],
            "minimum_direct_top1_at_256",
        )
        for seed in SEEDS
    )
    wins = sum(int(row["advances"]) for row in seed_results)
    passed = bool(
        wins >= 2
        and summaries["R2"]["score"] >= TARGET_SCORE
        and summaries["R2"]["precision"] >= PRECISION_FLOOR
        and summaries["R2"]["scored_positive_coverage"] >= COVERAGE_FLOOR
        and families["R2"]["lookup"]["macro_evidence_average_precision"]
        >= 0.95
        and retrieval_auroc >= 0.99
        and retrieval_top1 >= 0.95
        and latest_regression <= 0.01
        and corroboration_regression <= 0.01
    )
    return {
        "passed": passed,
        "seed_wins": wins,
        "seed_results": seed_results,
        "canonical_minimum_direct_auroc": retrieval_auroc,
        "canonical_minimum_direct_top1_at_256": retrieval_top1,
        "latest_valid_recall_regression": latest_regression,
        "corroboration_recall_regression": corroboration_regression,
    }


def _summarize(output_root: Path, stage: str) -> dict[str, Any]:
    results = _load_all(output_root, stage)
    arms, steps, scope, _ = _stage_settings(stage)
    summaries = {arm: _arm_summary(results, arm) for arm in arms}
    families = {arm: _family_summary(results, arm) for arm in arms}
    gate = (
        _screen_gate(results, summaries, families)
        if stage == "screen"
        else {"passed": summaries["R2"]["score"] >= TARGET_SCORE}
    )
    finalist = "R2" if gate["passed"] else "R0"
    payload = {
        "campaign": "Spider v0.7 canonical zero-shot binding",
        "stage": stage,
        "source_commit": _source_commit(),
        "dataset_hash": next(iter(results.values()))["dataset_hash"],
        "target_score": TARGET_SCORE,
        "score_definition": "min(exact_set, precision, recall)",
        "trainable_scope": scope,
        "completed_steps": steps,
        "arm_summaries": summaries,
        "per_family": families,
        "gate": gate,
        "selected_finalist": finalist,
        "accepted_training_run_count": len(arms) * len(SEEDS),
        "temperature_fit_count": 0,
        "symbol_overlap_count": 0,
        "sealed_access_count": 0,
        "run_source_commits": sorted(
            {row["source_commit"] for row in results.values()}
        ),
    }
    stage_root = output_root / stage
    _write(stage_root / "SUMMARY.json", payload)
    lines = [
        f"# Spider v0.7 {stage} results",
        "",
        "| Arm | Score | Exact set | Precision | Recall | Coverage | Macro AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
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
            f"Gate passed: `{gate['passed']}`.",
            f"Selected arm: `{finalist}`.",
            "No calibration or sealed evaluation was performed.",
        )
    )
    (stage_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    records = [
        _ledger_record(results[(arm, seed)], arm=arm, stage=stage)
        for arm in arms
        for seed in SEEDS
    ]
    _write_progress(stage_root, records)
    _write(
        stage_root / "FINALIST.json",
        {
            "selected_arm": finalist,
            "gate_passed": gate["passed"],
            "dataset_hash": payload["dataset_hash"],
            "score": summaries[finalist]["score"],
            "temperature_fit_count": 0,
            "symbol_overlap_count": 0,
            "sealed_access_count": 0,
        },
    )
    return payload


def _run_campaign(args: argparse.Namespace) -> None:
    arms, _, _, _ = _stage_settings(args.stage)
    if args.arm is not None and args.arm not in arms:
        raise ValueError(f"{args.arm} is not registered for {args.stage}")
    if args.stage == "confirm":
        screen = args.output_root / "screen" / "SUMMARY.json"
        if not screen.is_file() or not _load(screen)["gate"]["passed"]:
            raise RuntimeError("confirmation requires a passing frozen screen")
    selected_arms = (args.arm,) if args.arm else arms
    selected_seeds = (args.seed,) if args.seed else SEEDS
    if args.phase in {"run", "all"}:
        for arm in selected_arms:
            for seed in selected_seeds:
                metrics = _run_or_load(
                    arm=arm,
                    seed=seed,
                    stage=args.stage,
                    output_root=args.output_root,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=args.max_attempts,
                )
                _append_jsonl_once(
                    args.output_root / args.stage / "experiments.jsonl",
                    _ledger_record(metrics, arm=arm, stage=args.stage),
                )
        if args.arm is not None or args.seed is not None:
            print(
                json.dumps(
                    {
                        "completed_arms": list(selected_arms),
                        "completed_seeds": list(selected_seeds),
                        "summary_pending": True,
                    },
                    sort_keys=True,
                )
            )
            return
    print(json.dumps(_summarize(args.output_root, args.stage), sort_keys=True))


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
