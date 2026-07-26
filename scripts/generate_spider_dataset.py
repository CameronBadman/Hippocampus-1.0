#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from hippocampus.programs import (
    GraphProgramGenerator,
    build_split_manifest,
    default_split_specs,
    metadata_leakage_report,
    verify_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Spider v0 split manifests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spider_v0/splits"),
    )
    parser.add_argument("--case-scale", type=float, default=1.0)
    parser.add_argument("--leakage-cases-per-family", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = [
        build_split_manifest(spec)
        for spec in default_split_specs(case_scale=args.case_scale)
    ]
    for manifest in manifests:
        path = args.output / f"{manifest.spec.name}.json"
        path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        )

    generator = GraphProgramGenerator()
    diagnostic_cases = generator.generate_suite(
        cases_per_family=args.leakage_cases_per_family,
        seed=4401,
    )
    verification_errors = []
    for case in diagnostic_cases:
        report = verify_case(case)
        if not report.valid:
            verification_errors.append(
                {"case_id": case.case_id, "errors": report.errors}
            )
    leakage = metadata_leakage_report(diagnostic_cases)
    index = {
        "generator_version": manifests[0].spec.generator_version,
        "split_hashes": {
            manifest.spec.name: manifest.sha256 for manifest in manifests
        },
        "aggregate_sha256": hashlib.sha256(
            "".join(manifest.sha256 for manifest in manifests).encode()
        ).hexdigest(),
        "leakage_diagnostic": asdict(leakage),
        "verification_error_count": len(verification_errors),
        "verification_errors": verification_errors,
        "sealed_test_generated_but_not_opened": True,
    }
    (args.output / "MANIFEST_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(index, sort_keys=True))


if __name__ == "__main__":
    main()
