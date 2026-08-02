#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hippocampus.programs import (
    audit_aligned_program_labels,
    build_aligned_dev_manifest,
    default_aligned_dev_specs,
    generate_aligned_dev_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Spider v0.4 generator truth and observable labels."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("artifacts/spider_v0_4/splits"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/spider_v0_4/diagnostics/GENERATOR_LABEL_AUDIT.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_index = json.loads(
        (args.manifest_dir / "MANIFEST_INDEX.json").read_text()
    )
    partition_reports: dict[str, object] = {}
    all_cases = []
    for spec in default_aligned_dev_specs():
        cases = generate_aligned_dev_cases(spec)
        manifest = build_aligned_dev_manifest(spec, cases)
        committed = json.loads(
            (args.manifest_dir / f"{spec.name}.json").read_text()
        )
        if manifest.sha256 != committed["sha256"]:
            raise RuntimeError(f"{spec.name} manifest hash drift")
        report = audit_aligned_program_labels(cases)
        partition_reports[spec.name] = report.as_dict()
        all_cases.extend(cases)
    aggregate = audit_aligned_program_labels(all_cases)
    labels_consistent = (
        aggregate.invalid_case_count == 0
        and aggregate.evidence_label_mismatch_count == 0
        and aggregate.positive_summary_identity_rate == 1.0
        and aggregate.lookup_observable_rule_accuracy == 1.0
    )
    query_cardinality_advantage = (
        aggregate.query_cardinality_answerability_accuracy
        - aggregate.query_cardinality_majority_accuracy
    )
    payload = {
        "dataset_version": "spider-programs-v0.4-aligned-dev",
        "dataset_hash": manifest_index["aggregate_sha256"],
        "sealed_access_count": 0,
        "aggregate": aggregate.as_dict(),
        "partitions": partition_reports,
        "decision": {
            "supervisor_labels_mechanically_consistent": labels_consistent,
            "query_cardinality_answerability_advantage": (
                query_cardinality_advantage
            ),
            "unsupported_surface_codes_reused": (
                aggregate.unsupported_symbol_reuse_count > 0
            ),
            "dataset_interface_amendment_required": (
                query_cardinality_advantage > 0.05
                or aggregate.unsupported_case_count > 0
                and aggregate.unsupported_symbol_reuse_count == 0
            ),
            "positive_lookup_failure_explained_by_label_error": (
                not labels_consistent
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
