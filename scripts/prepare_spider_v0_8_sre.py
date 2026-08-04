#!/usr/bin/env python3
"""Validate and encode the non-sealed Spider v0.8 SRE corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hippocampus.sre import (  # noqa: E402
    DEFAULT_ENCODER,
    DEFAULT_ENCODER_REVISION,
    FrozenMiniLMEncoder,
    build_runtime_vocabulary,
    encode_sre_cases,
    load_encoded_cases,
    load_sre_cases,
    save_encoded_cases,
    split_sre_development,
)


DEFAULT_SOURCE = Path(
    "/home/cameron/projects/hippo-qwen-2/"
    "autoresearch/sre_incident_memory_v3/locked"
)
DEFAULT_DEMO = Path("/home/cameron/projects/hippo-qwen-2/demo_data/sre")
DEFAULT_CACHE = ROOT / "artifacts/spider_v0_8/local_rtx5070ti/cache"
DEFAULT_MODEL_PATH = Path(
    "/home/cameron/.cache/huggingface/hub/"
    "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/"
    f"{DEFAULT_ENCODER_REVISION}"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _balance(cases) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{case.scenario_family}|evidence={case.positive_count}"
                for case in cases
            ).items()
        )
    )


def _case_id_hash(cases) -> str:
    return hashlib.sha256(
        "\n".join(case.case_id for case in cases).encode()
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = json.loads(
        (ROOT / "artifacts/spider_v0_8/SOURCE_MANIFEST.json").read_text()
    )
    sources = {
        "train": (
            args.source_root / "train.inputs.jsonl",
            args.source_root / "train.labels.jsonl",
        ),
        "validation": (
            args.source_root / "validation.inputs.jsonl",
            args.source_root / "validation.labels.jsonl",
        ),
        "demo": (
            args.demo_root / "demo.inputs.jsonl",
            args.demo_root / "demo.labels.jsonl",
        ),
    }
    for _, paths in sources.items():
        for path in paths:
            expected = frozen["files"].get(path.name)
            if expected is None or _sha256(path) != expected:
                raise RuntimeError(f"frozen source hash disagrees for {path}")
    train = load_sre_cases(*sources["train"])
    validation = load_sre_cases(*sources["validation"])
    demo = load_sre_cases(*sources["demo"])
    selection, evaluation = split_sre_development(
        validation,
        selection_count=100,
    )
    partitions = {
        "train": train,
        "selection": selection,
        "evaluation": evaluation,
        "demo": demo,
    }
    vocabulary = build_runtime_vocabulary(train)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.cache_root / "vocabulary.json", asdict(vocabulary))

    pending = []
    for name, cases in partitions.items():
        path = args.cache_root / f"{name}.npz"
        if path.is_file() and not args.force:
            load_encoded_cases(path, cases=cases)
        else:
            pending.append((name, cases, path))
    if pending:
        encoder = FrozenMiniLMEncoder(
            model_name=DEFAULT_ENCODER,
            revision=DEFAULT_ENCODER_REVISION,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
        )
        for name, cases, path in pending:
            print(f"encoding {name}: {len(cases)} cases", flush=True)
            encoded = encode_sre_cases(
                cases,
                vocabulary=vocabulary,
                encoder=encoder,
            )
            save_encoded_cases(path, encoded)
            print(f"wrote {path} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)

    manifest = {
        "format": "spider-v0.8-sre-split-manifest-v1",
        "dataset": "sre-incident-memory-v3-synthetic",
        "encoder": DEFAULT_ENCODER,
        "encoder_revision": DEFAULT_ENCODER_REVISION,
        "sealed_access_count": 0,
        "selection_seed": 80_008,
        "partitions": {
            name: {
                "case_count": len(cases),
                "case_id_sha256": _case_id_hash(cases),
                "balance": _balance(cases),
            }
            for name, cases in partitions.items()
        },
    }
    _write_json(ROOT / "artifacts/spider_v0_8/SPLIT_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
