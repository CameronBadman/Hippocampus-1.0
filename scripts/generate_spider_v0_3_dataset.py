#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hippocampus.programs import (
    default_split_specs_v0_2,
    generate_split_cases,
)
from hippocampus.spider.protocol_v0_3 import (
    build_grouped_development_cases,
    verify_grouped_development_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate non-sealed Spider v0.3 development manifests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spider_v0_3/splits"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    specs = {spec.name: spec for spec in default_split_specs_v0_2()}
    train_spec = specs["train"]
    validation_spec = specs["validation_id"]
    if train_spec.sealed or validation_spec.sealed:
        raise RuntimeError("development generator resolved a sealed split")
    grouped = build_grouped_development_cases(
        generate_split_cases(train_spec),
        generate_split_cases(validation_spec),
    )
    verify_grouped_development_manifest(grouped)

    args.output.mkdir(parents=True, exist_ok=True)
    partitions = {
        "train": grouped.manifest.train,
        "development_calibration": grouped.manifest.calibration,
        "development_evaluation": grouped.manifest.evaluation,
    }
    for name, manifest in partitions.items():
        (args.output / f"{name}.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        )
    source_index = Path(
        "artifacts/spider_v0_1/splits/MANIFEST_INDEX.json"
    )
    index = {
        **grouped.manifest.to_dict(),
        "source_manifest_index": str(source_index),
        "source_manifest_index_sha256": _sha256(source_index),
        "sealed_cases_materialised": False,
        "sealed_manifest_loaded": False,
    }
    (args.output / "MANIFEST_INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(index, sort_keys=True))


if __name__ == "__main__":
    main()
