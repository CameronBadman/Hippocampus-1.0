#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hippocampus.programs import (
    V04_DATASET_VERSION,
    build_aligned_dev_manifest,
    default_aligned_dev_specs,
    generate_aligned_dev_cases,
    verify_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and hash the non-sealed Spider v0.4 partitions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spider_v0_4/splits"),
    )
    return parser.parse_args()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_payloads: list[dict[str, object]] = []
    all_case_ids: set[str] = set()
    all_base_case_ids: set[str] = set()
    for spec in default_aligned_dev_specs():
        if spec.sealed:
            raise RuntimeError("this command may not materialise sealed data")
        cases = generate_aligned_dev_cases(spec)
        for case in cases:
            report = verify_case(case)
            if not report.valid:
                raise RuntimeError(
                    f"invalid generated case {case.case_id}: {report.errors}"
                )
        manifest = build_aligned_dev_manifest(spec, cases)
        if not all_case_ids.isdisjoint(manifest.case_ids):
            raise RuntimeError("case identity leaked across v0.4 partitions")
        if not all_base_case_ids.isdisjoint(manifest.base_case_ids):
            raise RuntimeError("related base views crossed v0.4 partitions")
        all_case_ids.update(manifest.case_ids)
        all_base_case_ids.update(manifest.base_case_ids)
        payload = manifest.to_dict()
        destination = args.output_dir / f"{spec.name}.json"
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        split_payloads.append(
            {
                "name": spec.name,
                "case_count": len(cases),
                "sha256": manifest.sha256,
                "manifest_file": destination.name,
                "distributions": manifest.distributions,
            }
        )

    index_without_hash = {
        "dataset_version": V04_DATASET_VERSION,
        "partition_count": len(split_payloads),
        "total_case_count": len(all_case_ids),
        "sealed_access_count": 0,
        "related_view_partition_rule": "group by base_case_id",
        "stratification": {
            "independent_cycle": [
                "program_family",
                "answerable_or_unknown",
                "graph_size",
                "requested_path_length",
            ],
            "graph_size_buckets": [8, 16, 24, 32],
            "path_length_buckets": [1, 2, 3, 4],
            "structurally_coupled_observations": [
                "required_evidence_cardinality",
                "observed_path_length",
            ],
        },
        "splits": split_payloads,
    }
    index = {
        **index_without_hash,
        "aggregate_sha256": _canonical_sha256(index_without_hash),
    }
    (args.output_dir / "MANIFEST_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
