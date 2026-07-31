#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_3/evidence"
SEEDS = (1701, 1802, 1903)
ARMS = ("E0", "E1", "E2")
CONFIGS = {
    arm: ROOT / f"configs/spider_v0_3/evidence_{arm}.json"
    for arm in ARMS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the registered Spider v0.3 E0/E1/E2 matrix."
    )
    parser.add_argument(
        "--phase",
        choices=("screen", "full", "all", "summarize"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--screen-steps", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=21_600)
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


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _append_experiment_once(
    path: Path,
    record: dict[str, Any],
) -> None:
    existing_ids = (
        {
            json.loads(line)["experiment_id"]
            for line in path.read_text().splitlines()
            if line.strip()
        }
        if path.exists()
        else set()
    )
    if record["experiment_id"] not in existing_ids:
        _append_jsonl(path, record)


def _archive_incomplete_output(
    output_dir: Path,
    *,
    output_root: Path,
    experiment_id: str,
    source_commit: str,
) -> Path | None:
    """Preserve a partial attempt and return its exact resume checkpoint.

    Colab runtimes are preemptible. A restored Drive mirror can therefore
    contain a run directory without final metrics. Never overwrite that
    evidence: move it to a monotonically numbered recovery directory and
    resume from its latest atomic checkpoint when one exists.
    """

    recovery_root = output_root / "recovery"
    recovery_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        archived = recovery_root / (
            f"{experiment_id}-attempt-{attempt:03d}"
        )
        if not archived.exists():
            break
        attempt += 1
    output_dir.rename(archived)
    final_checkpoint = archived / "checkpoint.pt"
    periodic_checkpoints = sorted(
        archived.glob("checkpoint_step_*.pt")
    )
    checkpoint = (
        final_checkpoint
        if final_checkpoint.is_file()
        else periodic_checkpoints[-1]
        if periodic_checkpoints
        else None
    )
    _append_jsonl(
        output_root / "attempts.jsonl",
        {
            "experiment_id": experiment_id,
            "timestamp": _now(),
            "source_commit": source_commit,
            "status": "recovered",
            "archived_output": str(archived),
            "resume_checkpoint": (
                str(checkpoint) if checkpoint is not None else None
            ),
            "sealed_access_count": 0,
        },
    )
    return checkpoint


def _gate_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics["evidence_gate_metrics"].items()
    }


def _record_from_metrics(
    metrics: dict[str, Any],
    *,
    arm: str,
    phase: str,
    output_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": metrics["experiment_id"],
        "timestamp": metrics["timestamp"],
        "source_commit": metrics["source_commit"],
        "dataset_hash": metrics["dataset_hash"],
        "config_sha256": _sha256(config_path),
        "arm": arm,
        "phase": phase,
        "seed": metrics["resolved_seed"],
        "planned_steps": metrics["planned_steps"],
        "completed_steps": metrics["completed_steps"],
        "resumed_from_step": metrics["resumed_from_step"],
        "parameter_count": metrics["parameter_count"],
        "status": metrics["status"],
        "pass": metrics["pass"],
        "sealed_access_count": metrics["sealed_access_count"],
        "gate_metrics": _gate_metrics(metrics),
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "output_dir": str(output_dir),
        "failure_reason": metrics["failure_reason"],
    }


def _run_or_load(
    *,
    arm: str,
    seed: int,
    phase: str,
    output_root: Path,
    source_commit: str,
    stop_after_steps: int,
    timeout_seconds: int,
    precision_floor: float,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    suffix = "1k" if phase == "screen" else "6k"
    experiment_id = f"V03-{phase}-{arm}-s{seed}-{suffix}"
    output_dir = output_root / "runs" / experiment_id
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _load(metrics_path)
        if metrics.get("source_commit") != source_commit:
            raise RuntimeError(
                f"{experiment_id} belongs to a different source commit"
            )
        return metrics
    if output_dir.exists():
        recovered_checkpoint = _archive_incomplete_output(
            output_dir,
            output_root=output_root,
            experiment_id=experiment_id,
            source_commit=source_commit,
        )
        if recovered_checkpoint is not None:
            resume_checkpoint = recovered_checkpoint

    command = [
        sys.executable,
        str(ROOT / "scripts/spider_v0_3_evaluator.py"),
        "--config",
        str(CONFIGS[arm]),
        "--experiment-id",
        experiment_id,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--stop-after-steps",
        str(stop_after_steps),
        "--precision-floor",
        str(max(0.0, precision_floor)),
    ]
    if resume_checkpoint is not None:
        command.extend(("--resume-checkpoint", str(resume_checkpoint)))
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_id}.log"
    started = time.perf_counter()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        failure = {
            "experiment_id": experiment_id,
            "timestamp": _now(),
            "source_commit": source_commit,
            "arm": arm,
            "phase": phase,
            "seed": seed,
            "status": "crashed",
            "pass": False,
            "sealed_access_count": 0,
            "runtime_seconds": time.perf_counter() - started,
            "failure_reason": f"timeout after {exc.timeout} seconds",
            "output_dir": str(output_dir),
        }
        _append_jsonl(output_root / "attempts.jsonl", failure)
        raise RuntimeError(f"{experiment_id} timed out") from exc
    if not metrics_path.exists():
        tail = (
            log_path.read_text(errors="replace")[-20_000:]
            if log_path.exists()
            else ""
        )
        failure = {
            "experiment_id": experiment_id,
            "timestamp": _now(),
            "source_commit": source_commit,
            "arm": arm,
            "phase": phase,
            "seed": seed,
            "status": "crashed",
            "pass": False,
            "sealed_access_count": 0,
            "runtime_seconds": time.perf_counter() - started,
            "failure_reason": (
                f"exit {completed.returncode}; {tail}"
            )[-20_000:],
            "output_dir": str(output_dir),
        }
        _append_jsonl(output_root / "attempts.jsonl", failure)
        raise RuntimeError(f"{experiment_id} produced no metrics")
    return _load(metrics_path)


def _seed_gate(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float | bool]:
    recall_gain = candidate["recall"] - baseline["recall"]
    exact_gain = (
        candidate["exact_set_accuracy"]
        - baseline["exact_set_accuracy"]
    )
    precision_delta = candidate["precision"] - baseline["precision"]
    coverage_delta = (
        candidate["scored_positive_coverage"]
        - baseline["scored_positive_coverage"]
    )
    passed = (
        (recall_gain >= 0.05 or exact_gain >= 0.03)
        and precision_delta >= -0.02
        and coverage_delta >= -0.01
    )
    return {
        "passed": passed,
        "recall_gain": recall_gain,
        "exact_set_gain": exact_gain,
        "precision_delta": precision_delta,
        "scored_coverage_delta": coverage_delta,
    }


def _screen_decision(
    screen: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    eligible: list[str] = []
    for arm in ("E1", "E2"):
        seed_results: dict[str, dict[str, float | bool]] = {}
        for seed in SEEDS:
            baseline = screen[("E0", seed)]
            candidate = screen[(arm, seed)]
            result = _seed_gate(
                _gate_metrics(baseline),
                _gate_metrics(candidate),
            )
            result["passed"] = (
                bool(baseline["pass"])
                and bool(candidate["pass"])
                and bool(result["passed"])
            )
            seed_results[str(seed)] = result
        wins = sum(bool(result["passed"]) for result in seed_results.values())
        mean_recall_gain = sum(
            float(result["recall_gain"])
            for result in seed_results.values()
        ) / len(SEEDS)
        mean_exact_gain = sum(
            float(result["exact_set_gain"])
            for result in seed_results.values()
        ) / len(SEEDS)
        arm_eligible = wins >= 2
        if arm_eligible:
            eligible.append(arm)
        arms[arm] = {
            "eligible": arm_eligible,
            "seed_wins": wins,
            "mean_recall_gain": mean_recall_gain,
            "mean_exact_set_gain": mean_exact_gain,
            "seeds": seed_results,
        }
    winner = (
        max(
            eligible,
            key=lambda arm: (
                arms[arm]["mean_recall_gain"],
                arms[arm]["mean_exact_set_gain"],
                arm,
            ),
        )
        if eligible
        else None
    )
    return {
        "timestamp": _now(),
        "rule": "advance E0 and the best experimental arm passing >=2 seeds",
        "arms": arms,
        "experimental_winner": winner,
        "full_arms": ["E0", *([winner] if winner is not None else [])],
    }


def _final_evidence_selection(
    full: dict[tuple[str, int], dict[str, Any]],
    *,
    experimental_arm: str | None,
) -> dict[str, Any]:
    candidate_seed_gates: dict[str, dict[str, float | bool]] = {}
    full_candidate_passes = False
    if experimental_arm is not None:
        candidate_seed_gates = {
            str(seed): _seed_gate(
                _gate_metrics(full[("E0", seed)]),
                _gate_metrics(full[(experimental_arm, seed)]),
            )
            for seed in SEEDS
        }
        full_candidate_passes = (
            sum(
                bool(result["passed"])
                for result in candidate_seed_gates.values()
            )
            >= 2
        )
    selected_arm = (
        experimental_arm if full_candidate_passes else "E0"
    )
    selected_metrics = {
        seed: _gate_metrics(full[(selected_arm, seed)])
        for seed in SEEDS
    }
    median_exact = statistics.median(
        metrics["exact_set_accuracy"]
        for metrics in selected_metrics.values()
    )
    median_recall = statistics.median(
        metrics["recall"] for metrics in selected_metrics.values()
    )
    selected_seed = min(
        SEEDS,
        key=lambda seed: (
            abs(
                selected_metrics[seed]["exact_set_accuracy"]
                - median_exact
            )
            + abs(selected_metrics[seed]["recall"] - median_recall),
            seed,
        ),
    )
    selected = full[(selected_arm, selected_seed)]
    return {
        "timestamp": _now(),
        "arm_rule": (
            "candidate must pass the registered matched-seed gate again "
            "on full runs"
        ),
        "checkpoint_rule": (
            "minimum L1 distance to median exact-set accuracy and recall; "
            "ascending-seed tie break"
        ),
        "experimental_arm": experimental_arm,
        "experimental_full_seed_gates": candidate_seed_gates,
        "experimental_full_passed": full_candidate_passes,
        "selected_arm": selected_arm,
        "selected_seed": selected_seed,
        "selected_experiment_id": selected["experiment_id"],
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_evidence_threshold": selected["calibration"][
            "selected"
        ]["raw_probability_threshold"],
        "selected_metrics": selected_metrics[selected_seed],
        "selected_arm_medians": {
            "exact_set_accuracy": median_exact,
            "recall": median_recall,
        },
        "dataset_hash": selected["dataset_hash"],
        "source_commit": selected["source_commit"],
        "sealed_access_count": 0,
    }


def _write_summary(output_root: Path) -> None:
    ledger_path = output_root / "experiments.jsonl"
    records = (
        [
            json.loads(line)
            for line in ledger_path.read_text().splitlines()
            if line.strip()
        ]
        if ledger_path.exists()
        else []
    )
    lines = [
        "# Spider v0.3 evidence experiment ledger",
        "",
        "| Experiment | Phase | Arm | Seed | Recall | Exact set | "
        "Precision | Scored coverage | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        metrics = record.get("gate_metrics", {})
        lines.append(
            "| {experiment_id} | {phase} | {arm} | {seed} | "
            "{recall:.4f} | {exact:.4f} | {precision:.4f} | "
            "{coverage:.4f} | {status} |".format(
                experiment_id=record["experiment_id"],
                phase=record["phase"],
                arm=record["arm"],
                seed=record["seed"],
                recall=float(metrics.get("recall", 0.0)),
                exact=float(metrics.get("exact_set_accuracy", 0.0)),
                precision=float(metrics.get("precision", 0.0)),
                coverage=float(
                    metrics.get("scored_positive_coverage", 0.0)
                ),
                status=record["status"],
            )
        )
    decision_path = output_root / "SCREEN_DECISION.json"
    if decision_path.exists():
        decision = _load(decision_path)
        lines.extend(
            (
                "",
                "Registered screen decision: "
                f"`{decision['experimental_winner']}`.",
            )
        )
    selection_path = output_root / "FINAL_EVIDENCE_SELECTION.json"
    if selection_path.exists():
        selection = _load(selection_path)
        lines.extend(
            (
                "",
                "Frozen evidence finalist: "
                f"`{selection['selected_arm']}` seed "
                f"`{selection['selected_seed']}`.",
            )
        )
    (output_root / "EXPERIMENTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.screen_steps != 1000:
        raise ValueError("the registered screen length is exactly 1000 steps")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "summarize":
        _write_summary(output_root)
        return
    source_commit = _source_commit()
    manifest = _load(
        ROOT / "artifacts/spider_v0_3/splits/MANIFEST_INDEX.json"
    )
    protocol_record = {
        "source_commit": source_commit,
        "dataset_hash": manifest["aggregate_sha256"],
        "seeds": list(SEEDS),
        "screen_steps": args.screen_steps,
        "full_steps": 6000,
        "keep_policy": "pass_only",
        "sealed_access_allowed": False,
        "timestamp": _now(),
    }
    (output_root / "RUN_PROTOCOL.json").write_text(
        json.dumps(protocol_record, indent=2, sort_keys=True) + "\n"
    )

    screen: dict[tuple[str, int], dict[str, Any]] = {}
    if args.phase in {"screen", "all"}:
        for seed in SEEDS:
            baseline = _run_or_load(
                arm="E0",
                seed=seed,
                phase="screen",
                output_root=output_root,
                source_commit=source_commit,
                stop_after_steps=args.screen_steps,
                timeout_seconds=args.timeout_seconds,
                precision_floor=0.0,
            )
            screen[("E0", seed)] = baseline
            record = _record_from_metrics(
                baseline,
                arm="E0",
                phase="screen",
                output_dir=Path(baseline["checkpoint_path"]).parent,
                config_path=CONFIGS["E0"],
            )
            _append_experiment_once(
                output_root / "experiments.jsonl",
                record,
            )
            baseline_precision = float(
                baseline["calibration"]["selected"]["precision"]
            )
            for arm in ("E1", "E2"):
                metrics = _run_or_load(
                    arm=arm,
                    seed=seed,
                    phase="screen",
                    output_root=output_root,
                    source_commit=source_commit,
                    stop_after_steps=args.screen_steps,
                    timeout_seconds=args.timeout_seconds,
                    precision_floor=max(0.0, baseline_precision - 0.02),
                )
                screen[(arm, seed)] = metrics
                _append_experiment_once(
                    output_root / "experiments.jsonl",
                    _record_from_metrics(
                        metrics,
                        arm=arm,
                        phase="screen",
                        output_dir=Path(metrics["checkpoint_path"]).parent,
                        config_path=CONFIGS[arm],
                    ),
                )
        decision = _screen_decision(screen)
        (output_root / "SCREEN_DECISION.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n"
        )
    else:
        decision = _load(output_root / "SCREEN_DECISION.json")

    if args.phase in {"full", "all"}:
        winner = decision["experimental_winner"]
        full_arms = ("E0",) if winner is None else ("E0", str(winner))
        for seed in SEEDS:
            baseline_screen = _load(
                output_root
                / "runs"
                / f"V03-screen-E0-s{seed}-1k"
                / "metrics.json"
            )
            baseline_full = _run_or_load(
                arm="E0",
                seed=seed,
                phase="full",
                output_root=output_root,
                source_commit=source_commit,
                stop_after_steps=6000,
                timeout_seconds=args.timeout_seconds,
                precision_floor=0.0,
                resume_checkpoint=Path(
                    baseline_screen["checkpoint_path"]
                ),
            )
            _append_experiment_once(
                output_root / "experiments.jsonl",
                _record_from_metrics(
                    baseline_full,
                    arm="E0",
                    phase="full",
                    output_dir=Path(
                        baseline_full["checkpoint_path"]
                    ).parent,
                    config_path=CONFIGS["E0"],
                ),
            )
            if len(full_arms) == 1:
                continue
            baseline_precision = float(
                baseline_full["calibration"]["selected"]["precision"]
            )
            arm = full_arms[1]
            candidate_screen = _load(
                output_root
                / "runs"
                / f"V03-screen-{arm}-s{seed}-1k"
                / "metrics.json"
            )
            candidate_full = _run_or_load(
                arm=arm,
                seed=seed,
                phase="full",
                output_root=output_root,
                source_commit=source_commit,
                stop_after_steps=6000,
                timeout_seconds=args.timeout_seconds,
                precision_floor=max(0.0, baseline_precision - 0.02),
                resume_checkpoint=Path(
                    candidate_screen["checkpoint_path"]
                ),
            )
            _append_experiment_once(
                output_root / "experiments.jsonl",
                _record_from_metrics(
                    candidate_full,
                    arm=arm,
                    phase="full",
                    output_dir=Path(
                        candidate_full["checkpoint_path"]
                    ).parent,
                    config_path=CONFIGS[arm],
                ),
            )
        full = {
            (arm, seed): _load(
                output_root
                / "runs"
                / f"V03-full-{arm}-s{seed}-6k"
                / "metrics.json"
            )
            for arm in full_arms
            for seed in SEEDS
        }
        selection = _final_evidence_selection(
            full,
            experimental_arm=(
                str(winner) if winner is not None else None
            ),
        )
        (output_root / "FINAL_EVIDENCE_SELECTION.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n"
        )
    _write_summary(output_root)


if __name__ == "__main__":
    main()
