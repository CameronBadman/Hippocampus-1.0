#!/usr/bin/env python3
"""Run the gated Spider v0.4 learned evidence-set decoder comparison."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hippocampus.spider import load_experiment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_4/phase_f/local_rtx5070ti"
HISTORICAL_CONTROL = (
    ROOT / "artifacts/spider_v0_4/phase_d/local_rtx5070ti/full/runs"
)
SEEDS = (1701, 1802, 1903)
ARMS = ("F0", "F1", "F2", "F3")
TRAINED_ARMS = ("F1", "F2", "F3")
CONFIGS = {
    arm: ROOT / f"configs/spider_v0_4/phase_f_{arm}.json"
    for arm in ARMS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Spider v0.4 Phase F1 campaign."
    )
    parser.add_argument(
        "--phase",
        choices=("run", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--arm", choices=TRAINED_ARMS)
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
    """Prevent two orchestrators from mutating one campaign directory."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".campaign.lock"
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another Phase F orchestrator holds {lock_path}"
            ) from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _execute(
    command: list[str],
    *,
    log_path: Path,
    timeout_seconds: int,
    append: bool,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if append else "w") as log:
        return subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )


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
        "--stop-after-steps",
        "2000",
    ]


def _interrupted_stage(output_dir: Path) -> tuple[str, Path | None]:
    """Resolve the only valid next action for one interrupted run."""

    if (output_dir / "evaluation_pause.json").is_file():
        return "evaluation", None
    if (output_dir / "checkpoint.pt").is_file():
        return "selection", None
    partial = sorted(output_dir.glob("checkpoint_step_*.pt"))
    if partial:
        return "training", partial[-1]
    raise RuntimeError(f"incomplete run has no resumable checkpoint: {output_dir}")


def _run_or_load(
    *,
    arm: str,
    seed: int,
    output_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    experiment_id = f"V04-phase-F-{arm}-s{seed}"
    output_dir = output_root / "runs" / experiment_id
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = _load(metrics_path)
        if metrics["config_sha256"] != _sha256(CONFIGS[arm]):
            raise RuntimeError(f"{experiment_id} config hash has drifted")
        if metrics["sealed_access_count"] != 0:
            raise RuntimeError(f"{experiment_id} records sealed access")
        return metrics

    source_commit = _source_commit()
    command = _base_command(
        arm=arm,
        seed=seed,
        experiment_id=experiment_id,
        output_dir=output_dir,
    )
    log_path = output_root / "logs" / f"{experiment_id}.log"
    pause_path = output_dir / "evaluation_pause.json"
    try:
        if not output_dir.exists():
            completed = _execute(
                command + ["--pause-after-selection"],
                log_path=log_path,
                timeout_seconds=timeout_seconds,
                append=False,
            )
        elif not pause_path.is_file():
            stage, checkpoint = _interrupted_stage(output_dir)
            resume_arguments = (
                [
                    "--resume-training",
                    "--resume-checkpoint",
                    str(checkpoint),
                ]
                if stage == "training"
                else ["--resume-selection"]
            )
            completed = _execute(
                command
                + resume_arguments
                + [
                    "--pause-after-selection",
                    "--training-source-commit",
                    source_commit,
                    "--elapsed-before-seconds",
                    str(timeout_seconds),
                ],
                log_path=log_path,
                timeout_seconds=timeout_seconds,
                append=True,
            )
        else:
            completed = None
        if completed is not None and completed.returncode != 0:
            raise RuntimeError(
                f"{experiment_id} selection failed; inspect {log_path}"
            )
        if not pause_path.is_file():
            raise RuntimeError(f"{experiment_id} produced no evaluation pause")
        pause = _load(pause_path)
        training_source = str(pause["training_source_commit"])
        completed = _execute(
            command
            + [
                "--resume-evaluation",
                "--training-source-commit",
                training_source,
            ],
            log_path=log_path,
            timeout_seconds=timeout_seconds,
            append=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{experiment_id} evaluation failed; inspect {log_path}"
            )
    except (subprocess.TimeoutExpired, RuntimeError) as error:
        _append_once(
            output_root / "failed_experiments.jsonl",
            {
                "experiment_id": experiment_id,
                "timestamp": _now(),
                "source_commit": source_commit,
                "arm": arm,
                "seed": seed,
                "status": (
                    "timeout"
                    if isinstance(error, subprocess.TimeoutExpired)
                    else "crashed"
                ),
                "failure_reason": str(error),
                "sealed_access_count": 0,
            },
        )
        raise
    if not metrics_path.is_file():
        raise RuntimeError(f"{experiment_id} produced no metrics")
    return _load(metrics_path)


def _cardinality_error(metrics: dict[str, Any], family: str | None = None) -> float:
    cases = metrics["development_evaluation"]["evidence_pipeline"]["cases"]
    selected = [
        case for case in cases if family is None or case["family"] == family
    ]
    return statistics.mean(
        abs(
            int(case["predicted_cardinality"])
            - int(case["required_cardinality"])
        )
        for case in selected
    )


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(metrics)
    result["primary_metric"]["mean_absolute_cardinality_error"] = (
        _cardinality_error(result)
    )
    for family, values in result["per_family"].items():
        values["mean_absolute_cardinality_error"] = _cardinality_error(
            result, family
        )
    return result


def _historical_control(seed: int) -> dict[str, Any]:
    path = (
        HISTORICAL_CONTROL
        / f"V04-phase-D-D0-full-s{seed}"
        / "metrics.json"
    )
    metrics = _normalize_metrics(_load(path))
    if metrics["sealed_access_count"] != 0:
        raise RuntimeError("historical F0 control records sealed access")
    return metrics


def _verify_historical_control_equivalence() -> None:
    historical = load_experiment(
        ROOT / "configs/spider_v0_4/phase_d_D0.json"
    )
    registered = load_experiment(CONFIGS["F0"])
    fixed = (
        "schema",
        "query_dim",
        "model_config",
        "controller_config",
        "training_config",
        "loss_config",
        "device",
        "dtype",
    )
    drift = [
        name
        for name in fixed
        if getattr(historical, name) != getattr(registered, name)
    ]
    if drift:
        raise RuntimeError(f"F0 is not the historical D0 control: {drift}")
    if historical.raw["renderer"] != registered.raw["renderer"]:
        raise RuntimeError("F0 renderer differs from historical D0")
    if (
        historical.raw["dataset"]["aggregate_sha256"]
        != registered.raw["dataset"]["aggregate_sha256"]
    ):
        raise RuntimeError("F0 dataset differs from historical D0")


def _metric(row: dict[str, Any], name: str) -> float:
    return float(row["primary_metric"][name])


def _seed_gate(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        "exact_set": _metric(candidate, "exact_evidence_set_accuracy")
        - _metric(control, "exact_evidence_set_accuracy"),
        "recall": _metric(candidate, "recall") - _metric(control, "recall"),
        "precision": _metric(candidate, "precision")
        - _metric(control, "precision"),
        "cardinality_error": _metric(
            candidate, "mean_absolute_cardinality_error"
        )
        - _metric(control, "mean_absolute_cardinality_error"),
    }
    epsilon = 1e-12
    advances = (
        deltas["exact_set"] + epsilon >= 0.05
        and deltas["recall"] + epsilon >= 0.03
        and deltas["precision"] + epsilon >= -0.02
        and deltas["cardinality_error"] < -epsilon
    )
    return {"deltas": deltas, "advances": advances}


def _arm_summary(
    results: dict[tuple[str, int], dict[str, Any]], arm: str
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


def _family_summaries(
    results: dict[tuple[str, int], dict[str, Any]], arm: str
) -> dict[str, dict[str, float]]:
    families = tuple(sorted(results[(arm, SEEDS[0])]["per_family"]))
    fields = (
        "exact_evidence_set_accuracy",
        "precision",
        "recall",
        "macro_evidence_average_precision",
        "false_positives_per_case",
        "mean_absolute_cardinality_error",
    )
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


def _summary_key(summary: dict[str, float]) -> tuple[float, ...]:
    return (
        summary["constraint_seed_count"],
        summary["exact_evidence_set_accuracy"],
        summary["recall"],
        summary["macro_average_precision"],
        -summary["false_positives_per_case"],
        -summary["mean_absolute_cardinality_error"],
        -summary["mean_selected_step"],
    )


def _ledger_record(
    metrics: dict[str, Any], *, arm: str, reused: bool
) -> dict[str, Any]:
    return {
        "experiment_id": f"V04-phase-F-{arm}-s{metrics['resolved_seed']}",
        "timestamp": _now(),
        "source_commit": metrics["source_commit"],
        "source_experiment_id": metrics["experiment_id"] if reused else None,
        "dataset_hash": metrics["dataset_hash"],
        "config_sha256": _sha256(CONFIGS[arm]),
        "arm": arm,
        "seed": metrics["resolved_seed"],
        "status": "reused" if reused else metrics["status"],
        "parameter_count": metrics["parameter_count"],
        "planned_steps": metrics["planned_steps"],
        "completed_steps": metrics["completed_steps"],
        "selected_step": metrics["selected_step"],
        "primary_metric": metrics["primary_metric"],
        "per_family": metrics["per_family"],
        "runtime_seconds": 0.0 if reused else metrics["runtime_seconds"],
        "peak_cuda_memory_bytes": metrics["peak_cuda_memory_bytes"],
        "selected_checkpoint_sha256": metrics["selected_checkpoint_sha256"],
        "sealed_access_count": 0,
    }


def _load_all(output_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    results = {
        ("F0", seed): _historical_control(seed) for seed in SEEDS
    }
    for arm in TRAINED_ARMS:
        for seed in SEEDS:
            path = (
                output_root
                / "runs"
                / f"V04-phase-F-{arm}-s{seed}"
                / "metrics.json"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            results[(arm, seed)] = _normalize_metrics(_load(path))
    return results


def _summarize(output_root: Path) -> dict[str, Any]:
    results = _load_all(output_root)
    summaries = {arm: _arm_summary(results, arm) for arm in ARMS}
    gates: dict[str, Any] = {}
    advancing: list[str] = []
    for arm in TRAINED_ARMS:
        seed_rows = [
            {
                "seed": seed,
                **_seed_gate(results[("F0", seed)], results[(arm, seed)]),
            }
            for seed in SEEDS
        ]
        wins = sum(int(row["advances"]) for row in seed_rows)
        gates[f"{arm}_vs_F0"] = {
            "seed_results": seed_rows,
            "seed_wins": wins,
            "advances": wins >= 2,
        }
        if wins >= 2:
            advancing.append(arm)
    finalist = (
        max(advancing, key=lambda arm: _summary_key(summaries[arm]))
        if advancing
        else "F0"
    )
    payload = {
        "campaign": "Spider v0.4 Phase F1 learned set decoding",
        "source_commit": _source_commit(),
        "dataset_hash": next(iter(results.values()))["dataset_hash"],
        "arm_summaries": summaries,
        "per_family": {
            arm: _family_summaries(results, arm) for arm in ARMS
        },
        "gates": gates,
        "advancing_arms": advancing,
        "selected_finalist": finalist,
        "a100_replication_required": finalist != "F0",
        "new_training_run_count": len(TRAINED_ARMS) * len(SEEDS),
        "historical_control_reuse_count": len(SEEDS),
        "sealed_access_count": 0,
    }
    _write(output_root / "SUMMARY.json", payload)
    lines = [
        "# Spider v0.4 Phase F1 set-decoding results",
        "",
        "| Arm | Exact set | Precision | Recall | Coverage | Macro AP | MAE cardinality |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = summaries[arm]
        lines.append(
            f"| {arm} | {row['exact_evidence_set_accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['scored_positive_coverage']:.4f} | "
            f"{row['macro_average_precision']:.4f} | "
            f"{row['mean_absolute_cardinality_error']:.4f} |"
        )
    lines.extend(
        (
            "",
            f"Selected finalist: `{finalist}`.",
            "",
            "F0 reuses the immutable Phase D D0 full-run observations. "
            "No sealed split was accessed.",
        )
    )
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return payload


def _run_campaign(args: argparse.Namespace) -> None:
    _verify_historical_control_equivalence()
    _append_once(
        args.output_root / "experiments.jsonl",
        _ledger_record(_historical_control(SEEDS[0]), arm="F0", reused=True),
    )
    for seed in SEEDS[1:]:
        _append_once(
            args.output_root / "experiments.jsonl",
            _ledger_record(_historical_control(seed), arm="F0", reused=True),
        )

    if args.phase in {"run", "all"}:
        arms = (args.arm,) if args.arm else TRAINED_ARMS
        seeds = (args.seed,) if args.seed else SEEDS
        for arm in arms:
            for seed in seeds:
                metrics = _normalize_metrics(
                    _run_or_load(
                        arm=arm,
                        seed=seed,
                        output_root=args.output_root,
                        timeout_seconds=args.timeout_seconds,
                    )
                )
                _append_once(
                    args.output_root / "experiments.jsonl",
                    _ledger_record(metrics, arm=arm, reused=False),
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
    with _campaign_lock(args.output_root):
        _run_campaign(args)


if __name__ == "__main__":
    main()
