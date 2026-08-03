#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hippocampus.programs import (
    V06_DATASET_VERSION,
    audit_aligned_program_labels,
    build_aligned_dev_manifest,
    default_zero_shot_specs,
    generate_zero_shot_cases,
    observable_symbols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate symbol-disjoint Spider v0.6 development data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spider_v0_6/splits"),
    )
    return parser.parse_args()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _symbol_hash(symbols: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(symbols)).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    all_case_ids: set[str] = set()
    all_base_ids: set[str] = set()
    all_symbols: set[str] = set()
    for spec in default_zero_shot_specs():
        if spec.sealed:
            raise RuntimeError("this command may not materialise sealed data")
        cases = generate_zero_shot_cases(spec)
        symbols = observable_symbols(cases)
        if not all_symbols.isdisjoint(symbols):
            raise RuntimeError("surface symbol crossed v0.6 partitions")
        audit = audit_aligned_program_labels(cases)
        if audit.invalid_case_count or audit.evidence_label_mismatch_count:
            raise RuntimeError(f"{spec.name} supervisor audit failed")
        if audit.unsupported_case_count:
            raise RuntimeError(f"{spec.name} contains unsupported cases")
        if (
            audit.query_cardinality_answerability_accuracy
            != audit.query_cardinality_majority_accuracy
        ):
            raise RuntimeError(f"{spec.name} query cardinality leaks outcome")
        manifest = build_aligned_dev_manifest(spec, cases)
        if not all_case_ids.isdisjoint(manifest.case_ids):
            raise RuntimeError("case identity crossed v0.6 partitions")
        if not all_base_ids.isdisjoint(manifest.base_case_ids):
            raise RuntimeError("base identity crossed v0.6 partitions")
        all_case_ids.update(manifest.case_ids)
        all_base_ids.update(manifest.base_case_ids)
        all_symbols.update(symbols)
        destination = args.output_dir / f"{spec.name}.json"
        destination.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        summaries.append(
            {
                "name": spec.name,
                "case_count": len(cases),
                "sha256": manifest.sha256,
                "manifest_file": destination.name,
                "distributions": manifest.distributions,
                "symbol_count": len(symbols),
                "symbol_sha256": _symbol_hash(symbols),
                "query_cardinality_answerability_accuracy": (
                    audit.query_cardinality_answerability_accuracy
                ),
            }
        )
    base_index = {
        "dataset_version": V06_DATASET_VERSION,
        "purpose": "calibration-free candidate-relative evidence energy",
        "partition_count": len(summaries),
        "total_case_count": len(all_case_ids),
        "total_symbol_count": len(all_symbols),
        "symbol_overlap_count": 0,
        "sealed_access_count": 0,
        "related_view_partition_rule": "group by base_case_id",
        "splits": summaries,
    }
    index = {**base_index, "aggregate_sha256": _canonical_hash(base_index)}
    (args.output_dir / "MANIFEST_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
