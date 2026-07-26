#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from hippocampus.programs import (
    GeneratorConfig,
    GraphProgramGenerator,
    build_rollout_stress_manifest,
    build_split_manifest,
    default_split_specs_v0_2,
    metadata_leakage_report,
    verify_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Spider v0.2 manifests without opening sealed data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spider_v0_1/splits"),
    )
    parser.add_argument("--case-scale", type=float, default=1.0)
    parser.add_argument("--leakage-cases-per-family", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = default_split_specs_v0_2(case_scale=args.case_scale)
    manifests = tuple(build_split_manifest(spec) for spec in specs)
    for manifest in manifests:
        (args.output / f"{manifest.spec.name}.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        )

    stress_spec = next(
        spec for spec in specs if spec.name == "development_rollout_stress"
    )
    stress_manifest = build_rollout_stress_manifest(stress_spec)
    (args.output / "development_rollout_stress_states.json").write_text(
        json.dumps(stress_manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    generator = GraphProgramGenerator(
        GeneratorConfig(generator_version="spider-programs-v0.2")
    )
    diagnostic_cases = generator.generate_suite(
        cases_per_family=args.leakage_cases_per_family,
        seed=304_401,
    )
    verification_errors = [
        {
            "case_id": case.case_id,
            "errors": verify_case(case).errors,
        }
        for case in diagnostic_cases
        if not verify_case(case).valid
    ]
    leakage = metadata_leakage_report(diagnostic_cases)
    aggregate = hashlib.sha256(
        (
            "".join(manifest.sha256 for manifest in manifests)
            + stress_manifest.sha256
        ).encode()
    ).hexdigest()
    index = {
        "generator_version": "spider-programs-v0.2",
        "split_hashes": {
            manifest.spec.name: manifest.sha256
            for manifest in manifests
        },
        "rollout_stress_sha256": stress_manifest.sha256,
        "aggregate_sha256": aggregate,
        "leakage_diagnostic": asdict(leakage),
        "verification_error_count": len(verification_errors),
        "verification_errors": verification_errors,
        "sealed_manifest_generated": True,
        "sealed_cases_materialised": False,
        "historical_v0_artifacts_modified": False,
    }
    (args.output / "MANIFEST_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(index, sort_keys=True))


if __name__ == "__main__":
    main()
