#!/usr/bin/env python3
"""Run the preregistered Spider v0.4 evidence-readout matrix.

The screen trains every D arm to step 1,000.  Only the mechanically selected
pooled and Spider finalists resume to step 2,000.  The two stages share the
same 2,000-step schedule so a resumed run is exactly equivalent to an
uninterrupted run.
"""

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
DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_4/phase_d/local_rtx5070ti"
SEEDS = (1701, 1802, 1903)
ARMS = ("D0", "D1", "D2", "D3", "D4")
POOLED_ARMS = ("D0", "D1")
SPIDER_ARMS = ("D2", "D3", "D4")
CONFIGS = {
    arm: ROOT / f"configs/spider_v0_4/phase_d_{arm}.json"
    for arm in ARMS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the gated Spider v0.4 Phase D campaign."
    )
    parser.add_argument(
        "--phase",
        choices=("screen", "extend", "all", "summarize"),
        default="all",
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


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
    stage: str,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": metrics["experiment_id"],
        "timestamp": metrics["timestamp"],
        "source_commit": metrics["source_commit"],
        "dataset_hash": metrics["dataset_hash"],
        "config_sha256": _sha256(CONFIGS[arm]),
        "arm": arm,
        "stage": stage,
        "seed": metrics["resolved_seed"],
        "model_kind": (
            "pooled" if arm in POOLED_ARMS else "spider"
        ),
        "evidence_readout": {
            "D0": "shared",
            "D1": "dedicated_pooled",
            "D2": "shared",
            "D3": "dedicated_pooled",
            "D4": "slot_aware",
        }[arm],
        "parameter_count": metrics["parameter_count"],
        "planned_steps": metrics["planned_steps"],
        "completed_steps": metrics["completed_steps"],
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
    stage: str,
    output_root: Path,
    source_commit: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if stage not in {"screen", "full"}:
        raise ValueError("stage must be screen or full")
    experiment_id = f"V04-phase-D-{arm}-{stage}-s{seed}"
    output_dir = output_root / stage / "runs" / experiment_id
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _load(metrics_path)
        if (
            metrics["source_commit"] != source_commit
            and metrics.get("evaluation_source_commit") != source_commit
        ):
            raise RuntimeError(
                f"{experiment_id} was produced by another source commit"
            )
        return metrics

    base_command = [
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
        "--stop-after-steps",
        "1000" if stage == "screen" else "2000",
    ]
    if stage == "full":
        screen_dir = (
            output_root
            / "screen"
            / "runs"
            / f"V04-phase-D-{arm}-screen-s{seed}"
        )
        screen_checkpoint = screen_dir / "checkpoint.pt"
        if not (screen_dir / "metrics.json").is_file():
            raise FileNotFoundError(
                f"screen metrics are required before extension: {screen_dir}"
            )
        if not screen_checkpoint.is_file():
            raise FileNotFoundError(screen_checkpoint)
        base_command.extend(
            (
                "--resume-checkpoint",
                str(screen_checkpoint),
                "--prior-run",
                str(screen_dir),
            )
        )

    log_path = output_root / stage / "logs" / f"{experiment_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(command: list[str], *, append: bool) -> None:
        with log_path.open("a" if append else "w") as log:
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
                        "stage": stage,
                        "status": "timeout",
                        "timeout_seconds": timeout_seconds,
                        "failure_reason": str(error),
                        "sealed_access_count": 0,
                    },
                )
                raise
        if completed.returncode != 0:
            _append_once(
                output_root / "failed_experiments.jsonl",
                {
                    "experiment_id": experiment_id,
                    "timestamp": _now(),
                    "source_commit": source_commit,
                    "arm": arm,
                    "stage": stage,
                    "seed": seed,
                    "status": "crashed",
                    "returncode": completed.returncode,
                    "log_path": str(log_path),
                    "sealed_access_count": 0,
                },
            )
            raise RuntimeError(f"{experiment_id} failed; inspect {log_path}")

    if not output_dir.exists():
        execute(base_command + ["--pause-after-selection"], append=False)
        pause_path = output_dir / "evaluation_pause.json"
        if not pause_path.is_file():
            raise RuntimeError(
                f"{experiment_id} did not produce an evaluation pause"
            )
        training_source_commit = source_commit
    else:
        pause_path = output_dir / "evaluation_pause.json"
        if pause_path.is_file():
            training_source_commit = str(
                _load(pause_path)["training_source_commit"]
            )
        else:
            failures = (
                [
                    json.loads(line)
                    for line in (
                        output_root / "failed_experiments.jsonl"
                    ).read_text().splitlines()
                    if line.strip()
                ]
                if (output_root / "failed_experiments.jsonl").is_file()
                else []
            )
            matching = [
                failure
                for failure in failures
                if failure["experiment_id"] == experiment_id
            ]
            if not matching:
                raise RuntimeError(
                    f"incomplete run has no provenance: {output_dir}"
                )
            training_source_commit = str(matching[-1]["source_commit"])
            selection_command = base_command + [
                "--resume-selection",
                "--pause-after-selection",
                "--training-source-commit",
                training_source_commit,
                "--elapsed-before-seconds",
                str(timeout_seconds),
            ]
            execute(selection_command, append=True)
            if not pause_path.is_file():
                raise RuntimeError(
                    f"{experiment_id} did not finish resumed selection"
                )

    evaluation_command = base_command + [
        "--resume-evaluation",
        "--training-source-commit",
        training_source_commit,
    ]
    execute(evaluation_command, append=True)
    if not metrics_path.exists():
        raise RuntimeError(f"{experiment_id} produced no metrics")
    return _load(metrics_path)


def _metric(metrics: dict[str, Any], name: str) -> float:
    return float(metrics["primary_metric"][name])


def _seed_gate(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, object]:
    deltas = {
        "exact_set": _metric(
            candidate, "exact_evidence_set_accuracy"
        ) - _metric(control, "exact_evidence_set_accuracy"),
        "recall": _metric(candidate, "recall") - _metric(control, "recall"),
        "precision": _metric(candidate, "precision")
        - _metric(control, "precision"),
        "macro_average_precision": _metric(
            candidate, "macro_average_precision"
        ) - _metric(control, "macro_average_precision"),
    }
    advances = (
        (deltas["exact_set"] >= 0.05 or deltas["recall"] >= 0.07)
        and deltas["precision"] >= -0.02
        and deltas["macro_average_precision"] >= 0.0
    )
    return {"deltas": deltas, "advances": advances}


def _arm_gate(
    results: dict[tuple[str, int], dict[str, Any]],
    *,
    control_arm: str,
    candidate_arm: str,
) -> dict[str, object]:
    rows = [
        {
            "seed": seed,
            **_seed_gate(
                results[(control_arm, seed)],
                results[(candidate_arm, seed)],
            ),
        }
        for seed in SEEDS
    ]
    wins = sum(int(row["advances"]) for row in rows)
    return {
        "control": control_arm,
        "candidate": candidate_arm,
        "seed_results": rows,
        "seed_wins": wins,
        "advances": wins >= 2,
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
    )
    rows = [results[(arm, seed)] for seed in SEEDS]
    summary: dict[str, float] = {}
    for field in fields:
        values = [
            _metric(row, field)
            for row in rows
            if row["primary_metric"][field] is not None
        ]
        summary[field] = statistics.mean(values) if values else float("inf")
    summary["constraint_seed_count"] = float(
        sum(
            int(row["primary_metric"]["constraint_satisfied"])
            for row in rows
        )
    )
    summary["mean_selected_step"] = statistics.mean(
        float(row["selected_step"]) for row in rows
    )
    return summary


def _summary_key(summary: dict[str, float]) -> tuple[float, ...]:
    return (
        summary["constraint_seed_count"],
        summary["exact_evidence_set_accuracy"],
        summary["recall"],
        summary["macro_average_precision"],
        -summary["false_positives_per_case"],
        -summary["mean_worst_positive_rank"],
        -summary["mean_selected_step"],
    )


def _screen_decision(
    results: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, object]:
    gates = {
        "D1_vs_D0": _arm_gate(
            results, control_arm="D0", candidate_arm="D1"
        ),
        "D3_vs_D2": _arm_gate(
            results, control_arm="D2", candidate_arm="D3"
        ),
        "D4_vs_D2": _arm_gate(
            results, control_arm="D2", candidate_arm="D4"
        ),
        "D4_vs_D3": _arm_gate(
            results, control_arm="D3", candidate_arm="D4"
        ),
    }
    summaries = {arm: _arm_summary(results, arm) for arm in ARMS}
    pooled_candidates = ["D0"]
    if gates["D1_vs_D0"]["advances"]:
        pooled_candidates.append("D1")
    spider_candidates = ["D2"]
    for arm in ("D3", "D4"):
        if gates[f"{arm}_vs_D2"]["advances"]:
            spider_candidates.append(arm)
    pooled_finalist = max(
        pooled_candidates,
        key=lambda arm: _summary_key(summaries[arm]),
    )
    spider_finalist = max(
        spider_candidates,
        key=lambda arm: _summary_key(summaries[arm]),
    )
    return {
        "gates": gates,
        "arm_summaries": summaries,
        "pooled_finalist": pooled_finalist,
        "spider_finalist": spider_finalist,
        "full_run_arms": [pooled_finalist, spider_finalist],
        "sealed_access_count": 0,
    }


def _load_stage_results(
    output_root: Path,
    *,
    stage: str,
    arms: tuple[str, ...],
) -> dict[tuple[str, int], dict[str, Any]]:
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in arms:
        for seed in SEEDS:
            path = (
                output_root
                / stage
                / "runs"
                / f"V04-phase-D-{arm}-{stage}-s{seed}"
                / "metrics.json"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            results[(arm, seed)] = _load(path)
    return results


def _write_markdown_summary(
    path: Path,
    *,
    title: str,
    arms: tuple[str, ...],
    summaries: dict[str, dict[str, float]],
    decision: dict[str, object] | None = None,
) -> None:
    lines = [
        f"# {title}",
        "",
        "| Arm | Exact set | Precision | Recall | Coverage | Macro AP | FP/case |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        row = summaries[arm]
        lines.append(
            f"| {arm} | {row['exact_evidence_set_accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['scored_positive_coverage']:.4f} | "
            f"{row['macro_average_precision']:.4f} | "
            f"{row['false_positives_per_case']:.4f} |"
        )
    if decision is not None:
        lines.extend(
            (
                "",
                f"Pooled finalist: `{decision['pooled_finalist']}`.",
                f"Spider finalist: `{decision['spider_finalist']}`.",
            )
        )
    lines.extend(("", "No sealed split was accessed."))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _summarize_screen(output_root: Path) -> dict[str, object]:
    results = _load_stage_results(
        output_root,
        stage="screen",
        arms=ARMS,
    )
    decision = _screen_decision(results)
    payload = {
        "campaign": "Spider v0.4 Phase D readout screen",
        "source_commit": _source_commit(),
        "dataset_hash": next(iter(results.values()))["dataset_hash"],
        **decision,
    }
    _write(output_root / "SCREEN_SUMMARY.json", payload)
    _write(output_root / "ADVANCEMENT.json", decision)
    _write_markdown_summary(
        output_root / "SCREEN_SUMMARY.md",
        title="Spider v0.4 Phase D screen",
        arms=ARMS,
        summaries=decision["arm_summaries"],
        decision=decision,
    )
    return payload


def _summarize_full(output_root: Path) -> dict[str, object]:
    advancement = _load(output_root / "ADVANCEMENT.json")
    arms = tuple(str(arm) for arm in advancement["full_run_arms"])
    results = _load_stage_results(
        output_root,
        stage="full",
        arms=arms,
    )
    summaries = {arm: _arm_summary(results, arm) for arm in arms}
    payload = {
        "campaign": "Spider v0.4 Phase D finalist extension",
        "source_commit": _source_commit(),
        "dataset_hash": next(iter(results.values()))["dataset_hash"],
        "finalists": list(arms),
        "arm_summaries": summaries,
        "sealed_access_count": 0,
    }
    _write(output_root / "FULL_SUMMARY.json", payload)
    _write_markdown_summary(
        output_root / "FULL_SUMMARY.md",
        title="Spider v0.4 Phase D finalist extension",
        arms=arms,
        summaries=summaries,
    )
    return payload


def _record_result(
    *,
    metrics: dict[str, Any],
    arm: str,
    stage: str,
    output_root: Path,
) -> None:
    output_dir = (
        output_root
        / stage
        / "runs"
        / f"V04-phase-D-{arm}-{stage}-s{metrics['resolved_seed']}"
    )
    _append_once(
        output_root / "experiments.jsonl",
        _ledger_record(
            metrics,
            arm=arm,
            stage=stage,
            output_dir=output_dir,
        ),
    )


def main() -> None:
    args = parse_args()
    source_commit = _source_commit()
    selected_arms = (args.arm,) if args.arm else ARMS
    selected_seeds = (args.seed,) if args.seed else SEEDS

    if args.phase in {"screen", "all"}:
        for arm in selected_arms:
            for seed in selected_seeds:
                metrics = _run_or_load(
                    arm=arm,
                    seed=seed,
                    stage="screen",
                    output_root=args.output_root,
                    source_commit=source_commit,
                    timeout_seconds=args.timeout_seconds,
                )
                _record_result(
                    metrics=metrics,
                    arm=arm,
                    stage="screen",
                    output_root=args.output_root,
                )
        if args.arm is not None or args.seed is not None:
            print(
                json.dumps(
                    {
                        "completed_arms": list(selected_arms),
                        "completed_seeds": list(selected_seeds),
                        "screen_summary_pending": True,
                    },
                    sort_keys=True,
                )
            )
            return
        screen_summary = _summarize_screen(args.output_root)
    else:
        screen_summary = _summarize_screen(args.output_root)

    if args.phase in {"extend", "all"}:
        advancement = screen_summary
        finalists = tuple(str(arm) for arm in advancement["full_run_arms"])
        if args.arm is not None and args.arm not in finalists:
            raise ValueError(f"{args.arm} did not pass the screen gate")
        extension_arms = (args.arm,) if args.arm else finalists
        for arm in extension_arms:
            for seed in selected_seeds:
                metrics = _run_or_load(
                    arm=arm,
                    seed=seed,
                    stage="full",
                    output_root=args.output_root,
                    source_commit=source_commit,
                    timeout_seconds=args.timeout_seconds,
                )
                _record_result(
                    metrics=metrics,
                    arm=arm,
                    stage="full",
                    output_root=args.output_root,
                )
        if args.arm is not None or args.seed is not None:
            print(
                json.dumps(
                    {
                        "completed_arms": list(extension_arms),
                        "completed_seeds": list(selected_seeds),
                        "full_summary_pending": True,
                    },
                    sort_keys=True,
                )
            )
            return
        output = _summarize_full(args.output_root)
    elif args.phase == "summarize":
        full_summary = args.output_root / "FULL_SUMMARY.json"
        output = (
            _summarize_full(args.output_root)
            if full_summary.exists()
            else screen_summary
        )
    else:
        output = screen_summary
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
