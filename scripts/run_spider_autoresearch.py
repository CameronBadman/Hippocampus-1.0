#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts/spider_v0"
LEDGER = ARTIFACT_ROOT / "experiments.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered Spider v0 AutoResearch matrix."
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--train-cases", type=int, default=48)
    parser.add_argument("--eval-cases", type=int, default=8)
    parser.add_argument("--primary-seed", type=int, default=101)
    parser.add_argument(
        "--replicate-seeds",
        type=int,
        nargs="*",
        default=[202, 303],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--exploration-only", action="store_true")
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


def _controlled_config(
    source: str,
    *,
    experiment_id: str,
    mutation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = deepcopy(_load(source))
    config["name"] = experiment_id
    config["model"]["d_model"] = 64
    config["model"]["num_heads"] = 4
    config["model"]["path_rows"] = 4
    config["model"]["evidence_rows"] = 4
    config["model"]["dropout"] = 0.0
    config["controller"].update(
        {
            "max_rounds": 6,
            "frontier_width": 32,
            "hypotheses_per_node": 2,
            "context_read_budget": 4,
            "max_depth": 10,
            "search_budget": 4096,
        }
    )
    config["training"]["dtype"] = "float32"
    if mutation:
        for section, changes in mutation.items():
            config[section].update(changes)
    return config


def _experiment_matrix() -> list[dict[str, Any]]:
    return [
        {
            "id": "E001-pooled",
            "hypothesis": "Symmetric pooling is the minimum structural control.",
            "config": _controlled_config(
                "configs/spider_v0/baseline_pooled.json",
                experiment_id="E001-pooled",
            ),
        },
        {
            "id": "E002-flat",
            "hypothesis": (
                "A flat position-free Transformer tests whether explicit "
                "multi-set reads add value."
            ),
            "config": _controlled_config(
                "configs/spider_v0/baseline_flat_transformer.json",
                experiment_id="E002-flat",
            ),
        },
        {
            "id": "E003-recurrent-standard",
            "hypothesis": (
                "Tied multi-set recurrence improves path-length OOD over "
                "pooled and flat controls."
            ),
            "config": _controlled_config(
                "configs/spider_v0/recurrent_multiset.json",
                experiment_id="E003-recurrent-standard",
                mutation={"model": {"edge_mode": "standard"}},
            ),
        },
        {
            "id": "E004-recurrent-compositional",
            "hypothesis": (
                "Edge-conditioned value composition improves path-length OOD "
                "without harming equivalent-view behaviour."
            ),
            "config": _controlled_config(
                "configs/spider_v0/recurrent_compositional.json",
                experiment_id="E004-recurrent-compositional",
                mutation={
                    "model": {
                        "edge_mode": "compositional",
                        "edge_transforms": 4,
                        "adapter_rank": 8,
                    }
                },
            ),
        },
        {
            "id": "E005-untied",
            "hypothesis": (
                "Tied recurrence should retain ID accuracy while improving "
                "longer-path OOD relative to untied rounds."
            ),
            "config": _controlled_config(
                "configs/spider_v0/recurrent_multiset.json",
                experiment_id="E005-untied",
                mutation={
                    "model": {
                        "edge_mode": "standard",
                        "tied_recurrence": False,
                        "untied_rounds": 6,
                    }
                },
            ),
        },
        {
            "id": "E006-one-hypothesis",
            "hypothesis": (
                "Two hypotheses per node improve conflict handling over "
                "single-hypothesis pruning."
            ),
            "config": _controlled_config(
                "configs/spider_v0/recurrent_multiset.json",
                experiment_id="E006-one-hypothesis",
                mutation={
                    "model": {"edge_mode": "standard"},
                    "controller": {"hypotheses_per_node": 1},
                },
            ),
        },
        {
            "id": "E007-no-global-evidence",
            "hypothesis": (
                "The global evidence manifold improves corroboration/conflict "
                "decisions."
            ),
            "config": _controlled_config(
                "configs/spider_v0/recurrent_multiset.json",
                experiment_id="E007-no-global-evidence",
                mutation={
                    "model": {
                        "edge_mode": "standard",
                        "use_global_evidence": False,
                    }
                },
            ),
        },
        {
            "id": "E008-no-context-voi",
            "hypothesis": (
                "Value-of-information supervision reduces unnecessary reads "
                "without losing useful-read recall."
            ),
            "config": _controlled_config(
                "configs/spider_v0/recurrent_multiset.json",
                experiment_id="E008-no-context-voi",
                mutation={
                    "model": {"edge_mode": "standard"},
                    "controller": {"context_read_budget": 0},
                    "loss": {"context": 0.0, "context_cost": 0.0},
                },
            ),
        },
    ]


def _read_ledger() -> list[dict[str, Any]]:
    if not LEDGER.exists():
        return []
    return [
        json.loads(line)
        for line in LEDGER.read_text().splitlines()
        if line.strip()
    ]


def _append_record(record: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _run_guard_tests() -> None:
    subprocess.run(
        [str(ROOT / ".venv/bin/pytest"), "-q"],
        cwd=ROOT,
        check=True,
        timeout=300,
    )


def _run_experiment(
    definition: dict[str, Any],
    *,
    run_id: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config_dir = ARTIFACT_ROOT / "autoresearch/configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{run_id}.json"
    config = deepcopy(definition["config"])
    config["training"]["seed"] = seed
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    output_dir = ARTIFACT_ROOT / "autoresearch/runs" / run_id
    command = [
        "timeout",
        "5m",
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/spider_autoresearch_evaluator.py"),
        "--config",
        str(config_path),
        "--experiment-id",
        run_id,
        "--output-dir",
        str(output_dir),
        "--steps",
        str(args.steps),
        "--train-cases",
        str(args.train_cases),
        "--eval-cases",
        str(args.eval_cases),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=310,
        )
        result = json.loads(completed.stdout.splitlines()[-1])
        result["hypothesis"] = definition["hypothesis"]
        result["parent_experiment"] = definition["id"]
        result["command"] = command
        return result
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "crashed",
            "experiment_id": run_id,
            "parent_experiment": definition["id"],
            "hypothesis": definition["hypothesis"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "score": None,
            "failure_reason": str(exc),
            "stdout": getattr(exc, "stdout", None),
            "stderr": getattr(exc, "stderr", None),
            "command": command,
            "sealed_test_opened": False,
        }


def _write_summaries(records: list[dict[str, Any]]) -> None:
    accepted = [record for record in records if record["status"] == "accepted"]
    accepted.sort(key=lambda record: float(record["score"]), reverse=True)
    summary = [
        "# Spider v0 AutoResearch summary",
        "",
        "| Rank | Experiment | Seed | Score | Parameters | Runtime (s) | Status |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for rank, record in enumerate(accepted, start=1):
        summary.append(
            "| "
            f"{rank} | {record['experiment_id']} | {record['seed']} | "
            f"{float(record['score']):.4f} | "
            f"{record['parameter_count']} | "
            f"{float(record['runtime_seconds']):.1f} | "
            f"{record['status']} |"
        )
    failed = [record for record in records if record["status"] != "accepted"]
    if failed:
        summary.extend(("", "## Invalid or crashed runs", ""))
        for record in failed:
            summary.append(
                f"- `{record['experiment_id']}`: {record.get('failure_reason')}"
            )
    (ARTIFACT_ROOT / "EXPERIMENT_SUMMARY.md").write_text(
        "\n".join(summary) + "\n"
    )

    tsv = [
        "experiment_id\tparent\tseed\tstatus\tscore\tparameters\tsteps\truntime_seconds"
    ]
    for record in records:
        tsv.append(
            "\t".join(
                str(value)
                for value in (
                    record["experiment_id"],
                    record.get("parent_experiment", ""),
                    record.get("seed", ""),
                    record["status"],
                    record.get("score", ""),
                    record.get("parameter_count", ""),
                    record.get("training_steps", ""),
                    record.get("runtime_seconds", ""),
                )
            )
        )
    (ARTIFACT_ROOT / "results.tsv").write_text("\n".join(tsv) + "\n")

    points = [
        (20 + index * 70, 180 - 150 * float(record["score"]))
        for index, record in enumerate(accepted)
    ]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    width = max(120, 40 + len(points) * 70)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="210" '
        f'viewBox="0 0 {width} 210">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<line x1="20" y1="180" x2="'
        f'{width - 20}" y2="180" stroke="#999"/>'
        '<line x1="20" y1="20" x2="20" y2="180" stroke="#999"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#3857d6" '
        'stroke-width="2"/>'
        "</svg>\n"
    )
    (ARTIFACT_ROOT / "progress.svg").write_text(svg)


def _select_checkpoint(records: list[dict[str, Any]]) -> None:
    accepted = [record for record in records if record["status"] == "accepted"]
    if not accepted:
        return
    best = max(accepted, key=lambda record: float(record["score"]))
    manifest = {
        "experiment_id": best["experiment_id"],
        "parent_experiment": best.get("parent_experiment"),
        "score": best["score"],
        "seed": best["seed"],
        "config_path": best["config_path"],
        "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "source_commit": best["source_commit"],
        "dataset_split_hash": best["dataset_split_hash"],
        "sealed_test_opened": False,
        "tracked": False,
    }
    (ARTIFACT_ROOT / "selected_checkpoint.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    _run_guard_tests()
    definitions = _experiment_matrix()
    existing = _read_ledger()
    completed_ids = {record["experiment_id"] for record in existing}
    records = list(existing)

    for definition in definitions:
        run_id = definition["id"]
        if run_id in completed_ids:
            continue
        record = _run_experiment(
            definition,
            run_id=run_id,
            seed=args.primary_seed,
            args=args,
        )
        _append_record(record)
        records.append(record)
        _write_summaries(records)

    if not args.exploration_only:
        exploration = [
            record
            for record in records
            if record["status"] == "accepted"
            and str(record["experiment_id"]).startswith("E")
        ]
        best_parents = []
        for record in sorted(
            exploration,
            key=lambda item: float(item["score"]),
            reverse=True,
        ):
            parent = str(record["parent_experiment"])
            if parent not in best_parents:
                best_parents.append(parent)
            if len(best_parents) == 2:
                break
        by_id = {definition["id"]: definition for definition in definitions}
        for parent in best_parents:
            for seed in args.replicate_seeds:
                run_id = f"F-{parent}-s{seed}"
                if run_id in completed_ids:
                    continue
                record = _run_experiment(
                    by_id[parent],
                    run_id=run_id,
                    seed=seed,
                    args=args,
                )
                _append_record(record)
                records.append(record)
                _write_summaries(records)

    _run_guard_tests()
    _write_summaries(records)
    _select_checkpoint(records)
    accepted_count = sum(record["status"] == "accepted" for record in records)
    print(
        json.dumps(
            {
                "status": "completed",
                "accepted_experiments": accepted_count,
                "ledger": str(LEDGER),
                "summary": str(ARTIFACT_ROOT / "EXPERIMENT_SUMMARY.md"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
