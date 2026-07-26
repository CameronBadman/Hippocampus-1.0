#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hippocampus.programs import (
    RECURRENCE_DATASET_VERSION,
    build_recurrence_necessity_manifest,
    default_recurrence_necessity_specs,
    generate_recurrence_necessity_cases,
    recurrence_metadata_leakage_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Spider recurrence-necessity development manifests."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/spider_v0_2/splits"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, object] = {}
    specs = default_recurrence_necessity_specs()
    for spec in specs:
        manifest = build_recurrence_necessity_manifest(spec)
        payload = manifest.as_dict()
        manifests[spec.name] = payload
        (args.output_dir / f"{spec.name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )

    validation_spec = next(
        spec
        for spec in specs
        if spec.name == "validation_recurrence_necessity"
    )
    leakage = recurrence_metadata_leakage_report(
        generate_recurrence_necessity_cases(validation_spec)
    )
    if leakage.final_position_accuracy >= 0.45:
        raise RuntimeError("metadata-only position heuristic exceeded leakage guard")
    if leakage.first_hop_profile_mismatch_count:
        raise RuntimeError("first-hop local profiles are not fully matched")
    index = {
        "dataset_version": RECURRENCE_DATASET_VERSION,
        "sealed_split_present": False,
        "manifests": manifests,
        "leakage": leakage.as_dict(),
    }
    (args.output_dir / "MANIFEST_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "dataset_version": RECURRENCE_DATASET_VERSION,
                "manifest_hashes": {
                    name: payload["sha256"]
                    for name, payload in manifests.items()
                },
                "leakage": leakage.as_dict(),
                "sealed_split_present": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
