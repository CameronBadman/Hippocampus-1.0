#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from hippocampus.programs import (
    V07_DATASET_VERSION,
    ProgramFamily,
    audit_aligned_program_labels,
    build_aligned_dev_manifest,
    default_binding_specs,
    generate_binding_cases,
    observable_symbols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate symbol-disjoint Spider v0.7 binding data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spider_v0_7/splits"),
    )
    return parser.parse_args()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _symbol_hash(symbols: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(symbols)).encode()).hexdigest()


def _scalar_inventory(case) -> Counter[float]:
    return Counter(
        atom.scalar
        for edge in case.edges
        for atom in edge.atoms
        if atom.scalar is not None
    )


def _validate_lookup_pairs(cases) -> int:
    grouped: dict[int, list] = {}
    for case in cases:
        if case.family is ProgramFamily.LOOKUP:
            grouped.setdefault(case.seed, []).append(case)
    pair_count = 0
    for seed, pair in grouped.items():
        if len(pair) != 2 or {case.answerable for case in pair} != {False, True}:
            raise RuntimeError(f"lookup seed {seed} is not a matched pair")
        positive = next(case for case in pair if case.answerable)
        negative = next(case for case in pair if not case.answerable)
        if positive.query_atoms != negative.query_atoms:
            raise RuntimeError(f"lookup seed {seed} query drifted")
        if observable_symbols((positive,)) != observable_symbols((negative,)):
            raise RuntimeError(f"lookup seed {seed} symbol inventory drifted")
        if _scalar_inventory(positive) != _scalar_inventory(negative):
            raise RuntimeError(f"lookup seed {seed} scalar inventory drifted")
        positive_shape = (
            tuple(len(node.summary_atoms) for node in positive.nodes),
            tuple(
                (edge.source_node, edge.destination_node, len(edge.atoms))
                for edge in positive.edges
            ),
        )
        negative_shape = (
            tuple(len(node.summary_atoms) for node in negative.nodes),
            tuple(
                (edge.source_node, edge.destination_node, len(edge.atoms))
                for edge in negative.edges
            ),
        )
        if positive_shape != negative_shape:
            raise RuntimeError(f"lookup seed {seed} metadata drifted")
        pair_count += 1
    return pair_count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    all_case_ids: set[str] = set()
    all_base_ids: set[str] = set()
    all_symbols: set[str] = set()
    total_pairs = 0
    for spec in default_binding_specs():
        if spec.sealed:
            raise RuntimeError("this command may not materialise sealed data")
        cases = generate_binding_cases(spec)
        symbols = observable_symbols(cases)
        if not all_symbols.isdisjoint(symbols):
            raise RuntimeError("surface symbol crossed v0.7 partitions")
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
        pair_count = _validate_lookup_pairs(cases)
        total_pairs += pair_count
        manifest = build_aligned_dev_manifest(spec, cases)
        if not all_case_ids.isdisjoint(manifest.case_ids):
            raise RuntimeError("case identity crossed v0.7 partitions")
        if not all_base_ids.isdisjoint(manifest.base_case_ids):
            raise RuntimeError("base identity crossed v0.7 partitions")
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
                "matched_lookup_pair_count": pair_count,
                "query_cardinality_answerability_accuracy": (
                    audit.query_cardinality_answerability_accuracy
                ),
            }
        )
    base_index = {
        "dataset_version": V07_DATASET_VERSION,
        "purpose": "zero-shot relation/value binding and exact evidence",
        "partition_count": len(summaries),
        "total_case_count": len(all_case_ids),
        "total_symbol_count": len(all_symbols),
        "symbol_overlap_count": 0,
        "matched_lookup_pair_count": total_pairs,
        "sealed_access_count": 0,
        "related_view_partition_rule": "matched lookup seeds stay in one partition",
        "splits": summaries,
    }
    index = {**base_index, "aggregate_sha256": _canonical_hash(base_index)}
    (args.output_dir / "MANIFEST_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
