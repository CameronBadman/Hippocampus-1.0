#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from hippocampus import GraphSchema
from hippocampus.programs import (
    IdentifiabilityProbeConfig,
    SyntheticManifoldRenderer,
    run_renderer_identifiability,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_4/renderer_identifiability"
GEOMETRIES = (
    ("A0", "independent"),
    ("A1", "shared_additive"),
    ("A2", "orthogonal_aligned"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Spider v0.4 cross-modal renderer identity."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-symbols", type=int, default=4096)
    parser.add_argument("--test-symbols", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _renderer_hash() -> str:
    paths = (
        ROOT / "src/hippocampus/programs/renderer.py",
        ROOT / "src/hippocampus/programs/identifiability.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse {output_root}")
    output_root.mkdir(parents=True)
    schema = GraphSchema(summary_dim=16, context_dim=16, edge_dim=16)
    config = IdentifiabilityProbeConfig(
        train_symbol_count=args.train_symbols,
        test_symbol_count=args.test_symbols,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    source_commit = _source_commit()
    records: list[dict[str, object]] = []
    for arm, geometry in GEOMETRIES:
        started = time.perf_counter()
        renderer = SyntheticManifoldRenderer(
            schema,
            query_dim=16,
            seed=91_337,
            geometry=geometry,
        )
        report = run_renderer_identifiability(renderer, config=config)
        record = {
            "experiment_id": f"V04-renderer-{arm}-s{args.seed}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_commit": source_commit,
            "renderer_version": renderer.renderer_version,
            "renderer_hash": _renderer_hash(),
            "arm": arm,
            "geometry": geometry,
            "probe_seed": args.seed,
            "status": "accepted",
            "sealed_access_count": 0,
            "runtime_seconds": time.perf_counter() - started,
            "report": report.as_dict(),
        }
        records.append(record)
        (output_root / f"{arm}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    with (output_root / "experiments.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    by_arm = {str(record["arm"]): record for record in records}
    a0 = by_arm["A0"]["report"]
    a1 = by_arm["A1"]["report"]
    a2 = by_arm["A2"]["report"]
    assert isinstance(a0, dict) and isinstance(a1, dict) and isinstance(a2, dict)
    decision = {
        "source_commit": source_commit,
        "renderer_hash": _renderer_hash(),
        "A0_near_chance": (
            float(a0["macro_auroc"]) < 0.60
            and float(a0["macro_top1_at_64"]) < 0.10
        ),
        "A1_upper_bound_passed": bool(a1["passed"]),
        "A2_gate_passed": bool(a2["passed"]),
        "graph_training_unlocked": bool(a2["passed"]),
        "sealed_access_count": 0,
    }
    (output_root / "GATE_DECISION.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Spider v0.4 renderer identifiability",
        "",
        "| Arm | Geometry | AUROC | Top-1@64 | Top-1@256 | Gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        report = record["report"]
        assert isinstance(report, dict)
        lines.append(
            "| {arm} | {geometry} | {auroc:.4f} | {top64:.4f} | "
            "{top256:.4f} | {gate} |".format(
                arm=record["arm"],
                geometry=record["geometry"],
                auroc=float(report["minimum_auroc"]),
                top64=float(report["minimum_top1_at_64"]),
                top256=float(report["minimum_top1_at_256"]),
                gate="pass" if report["passed"] else "fail",
            )
        )
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, sort_keys=True))
    if not (
        decision["A0_near_chance"]
        and decision["A1_upper_bound_passed"]
        and decision["A2_gate_passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
