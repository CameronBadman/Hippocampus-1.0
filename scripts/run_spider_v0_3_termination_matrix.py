#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1701, 1802, 1903)
ARMS = ("T0", "T1", "T2")
TERMINATION_CONFIGS = {
    arm: ROOT / f"configs/spider_v0_3/termination_{arm}.json"
    for arm in ARMS
}
EVIDENCE_CONFIGS = {
    arm: ROOT / f"configs/spider_v0_3/evidence_{arm}.json"
    for arm in ("E0", "E1", "E2")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the registered Spider v0.3 T0/T1/T2 matrix."
    )
    parser.add_argument("--evidence-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=21_600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


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
    if record["experiment_id"] not in existing:
        _append_jsonl(path, record)


def _metrics_view(metrics: dict[str, Any]) -> dict[str, float]:
    state = metrics["state_evaluation"]
    autonomous = metrics["autonomous_evaluation"]
    return {
        "continuation_recall": float(state["continuation_recall"]),
        "premature_stop_rate": float(state["premature_stop_rate"]),
        "unknown_macro_recall": float(state["unknown_macro_recall"]),
        "autonomous_success": float(
            autonomous["primary_autonomous_success"]
        ),
        "fixed_horizon_success": float(
            autonomous["fixed_horizon_structural_success"]
        ),
        "autonomous_retention": float(metrics["autonomous_retention"]),
        "risk_among_answered": float(
            autonomous["rollout"]["risk_among_answered"]
        ),
        "false_answer_rate": float(
            autonomous["rollout"]["false_answer_rate"]
        ),
        "evidence_recall": float(autonomous["evidence"]["recall"]),
        "exact_evidence_set_accuracy": float(
            autonomous["evidence_pipeline"]["overall"][
                "exact_evidence_set_accuracy"
            ]
        ),
    }


def _seed_gate(
    metrics: dict[str, Any],
    *,
    baseline_risk: float,
) -> dict[str, float | bool]:
    values = _metrics_view(metrics)
    risk_delta = values["risk_among_answered"] - baseline_risk
    passed = (
        bool(metrics["pass"])
        and values["continuation_recall"] >= 0.95
        and values["premature_stop_rate"] < 0.25
        and values["autonomous_retention"] >= 0.85
        and values["unknown_macro_recall"] >= 0.70
        and risk_delta <= 0.02
    )
    return {
        "passed": passed,
        "risk_delta_vs_T0": risk_delta,
        **values,
    }


def _arm_decision(
    runs: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    arm_records: dict[str, Any] = {}
    eligible: list[str] = []
    for arm in ARMS:
        gates = {
            str(seed): _seed_gate(
                runs[(arm, seed)],
                baseline_risk=_metrics_view(runs[("T0", seed)])[
                    "risk_among_answered"
                ],
            )
            for seed in SEEDS
        }
        passes = sum(bool(gate["passed"]) for gate in gates.values())
        mean_autonomous = sum(
            float(gate["autonomous_success"])
            for gate in gates.values()
        ) / len(SEEDS)
        mean_retention = sum(
            float(gate["autonomous_retention"])
            for gate in gates.values()
        ) / len(SEEDS)
        is_eligible = passes >= 2
        if is_eligible:
            eligible.append(arm)
        arm_records[arm] = {
            "eligible": is_eligible,
            "seed_passes": passes,
            "mean_autonomous_success": mean_autonomous,
            "mean_autonomous_retention": mean_retention,
            "seeds": gates,
        }
    selected = (
        max(
            eligible,
            key=lambda arm: (
                arm_records[arm]["mean_autonomous_success"],
                arm_records[arm]["mean_autonomous_retention"],
                -ARMS.index(arm),
            ),
        )
        if eligible
        else None
    )
    return {
        "timestamp": _now(),
        "rule": (
            "at least two seed-level absolute gate passes; maximize mean "
            "autonomous success, then retention, then simpler arm"
        ),
        "arms": arm_records,
        "selected_arm": selected,
        "termination_gate_passed": selected is not None,
        "sealed_access_count": 0,
    }


def _archive_incomplete(
    output_dir: Path,
    *,
    output_root: Path,
    experiment_id: str,
) -> None:
    recovery = output_root / "recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (
        recovery / f"{experiment_id}-attempt-{attempt:03d}"
    ).exists():
        attempt += 1
    output_dir.rename(
        recovery / f"{experiment_id}-attempt-{attempt:03d}"
    )


def _run_or_load(
    *,
    arm: str,
    seed: int,
    evidence_run_dir: Path,
    evidence_config: Path,
    output_root: Path,
    timeout_seconds: int,
    device: str,
    source_commit: str,
) -> dict[str, Any]:
    experiment_id = f"V03-termination-{arm}-s{seed}"
    output_dir = output_root / "runs" / experiment_id
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _load(metrics_path)
        if metrics["source_commit"] != source_commit:
            raise RuntimeError(
                f"{experiment_id} belongs to another source commit"
            )
        return metrics
    if output_dir.exists():
        _archive_incomplete(
            output_dir,
            output_root=output_root,
            experiment_id=experiment_id,
        )
    command = [
        sys.executable,
        str(ROOT / "scripts/train_spider_v0_3_termination.py"),
        "--evidence-run-dir",
        str(evidence_run_dir),
        "--evidence-config",
        str(evidence_config),
        "--config",
        str(TERMINATION_CONFIGS[arm]),
        "--experiment-id",
        experiment_id,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_id}.log"
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
            "arm": arm,
            "seed": seed,
            "status": "crashed",
            "failure_reason": f"timeout after {exc.timeout} seconds",
            "sealed_access_count": 0,
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
            "arm": arm,
            "seed": seed,
            "status": "crashed",
            "failure_reason": (
                f"exit {completed.returncode}; {tail}"
            )[-20_000:],
            "sealed_access_count": 0,
        }
        _append_jsonl(output_root / "attempts.jsonl", failure)
        raise RuntimeError(f"{experiment_id} produced no metrics")
    return _load(metrics_path)


def _record(
    metrics: dict[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    return {
        "experiment_id": metrics["experiment_id"],
        "timestamp": metrics["timestamp"],
        "source_commit": metrics["source_commit"],
        "dataset_hash": metrics["dataset_hash"],
        "arm": arm,
        "seed": metrics["seed"],
        "status": metrics["status"],
        "pass": metrics["pass"],
        "sealed_access_count": metrics["sealed_access_count"],
        "metrics": _metrics_view(metrics),
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
        "checkpoint_path": metrics["checkpoint_path"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "source_evidence": metrics["source_evidence"],
    }


def _write_summary(output_root: Path) -> None:
    records = [
        json.loads(line)
        for line in (output_root / "experiments.jsonl").read_text().splitlines()
        if line.strip()
    ]
    lines = [
        "# Spider v0.3 termination experiment ledger",
        "",
        "| Experiment | Arm | Seed | Continue recall | Premature stop | "
        "Autonomous | Retention | Unknown macro | Risk |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            "| {experiment_id} | {arm} | {seed} | {continue_:.4f} | "
            "{premature:.4f} | {autonomous:.4f} | {retention:.4f} | "
            "{unknown:.4f} | {risk:.4f} |".format(
                experiment_id=record["experiment_id"],
                arm=record["arm"],
                seed=record["seed"],
                continue_=metrics["continuation_recall"],
                premature=metrics["premature_stop_rate"],
                autonomous=metrics["autonomous_success"],
                retention=metrics["autonomous_retention"],
                unknown=metrics["unknown_macro_recall"],
                risk=metrics["risk_among_answered"],
            )
        )
    decision_path = output_root / "TERMINATION_DECISION.json"
    if decision_path.exists():
        decision = _load(decision_path)
        lines.extend(
            (
                "",
                f"Selected arm: `{decision['selected_arm']}`.",
                "",
                "Multi-binding architecture experiments are permitted only "
                "when `termination_gate_passed` is true.",
            )
        )
    (output_root / "EXPERIMENTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root = args.evidence_output_root.resolve()
    selection = _load(evidence_root / "FINAL_EVIDENCE_SELECTION.json")
    if selection["sealed_access_count"] != 0:
        raise RuntimeError("selected evidence model accessed sealed data")
    evidence_arm = str(selection["selected_arm"])
    evidence_config = EVIDENCE_CONFIGS[evidence_arm]
    evidence_run_dir = Path(
        selection["selected_checkpoint_path"]
    ).resolve().parent
    if not evidence_run_dir.is_dir():
        raise FileNotFoundError("selected evidence run directory is missing")
    source_commit = _source_commit()
    protocol = {
        "source_commit": source_commit,
        "dataset_hash": selection["dataset_hash"],
        "evidence_selection": selection,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "device": args.device,
        "sealed_access_allowed": False,
        "timestamp": _now(),
    }
    (output_root / "RUN_PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in SEEDS:
        for arm in ARMS:
            metrics = _run_or_load(
                arm=arm,
                seed=seed,
                evidence_run_dir=evidence_run_dir,
                evidence_config=evidence_config,
                output_root=output_root,
                timeout_seconds=args.timeout_seconds,
                device=args.device,
                source_commit=source_commit,
            )
            runs[(arm, seed)] = metrics
            _append_once(
                output_root / "experiments.jsonl",
                _record(metrics, arm=arm),
            )
    decision = _arm_decision(runs)
    (output_root / "TERMINATION_DECISION.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    _write_summary(output_root)


if __name__ == "__main__":
    main()
