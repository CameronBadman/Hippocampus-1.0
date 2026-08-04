#!/usr/bin/env python3
"""Evaluate a frozen Spider v0.8 checkpoint on a non-sealed dev partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hippocampus.sre import (  # noqa: E402
    PackedSRECanonicalRetriever,
    SREModelConfig,
    evaluate_model,
    load_development_corpus,
    model_order_invariance,
)


DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_8/local_rtx5070ti"
DEFAULT_MANIFEST = ROOT / "artifacts/spider_v0_8/SELECTED_CHECKPOINT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--partition",
        choices=("selection", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_model(manifest_path: Path, device: torch.device):
    manifest = json.loads(manifest_path.read_text())
    if manifest["sealed_access_count"] != 0:
        raise RuntimeError("refusing a checkpoint associated with sealed access")
    config = json.loads((ROOT / manifest["config"]).read_text())
    if manifest["arm"] != config["arm"]:
        raise ValueError("result and selected model configuration disagree")
    weights_path = ROOT / manifest["portable_weights_path"]
    if _sha256(weights_path) != manifest["portable_weights_sha256"]:
        raise RuntimeError("portable selected weights hash disagrees")
    model = PackedSRECanonicalRetriever(SREModelConfig(**config["model"])).to(device)
    model.load_state_dict(load_file(weights_path, device=str(device)))
    return model, manifest


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    corpus = load_development_corpus(cache_root=args.output_root / "cache")
    partition = getattr(corpus, args.partition)
    model, manifest = load_selected_model(args.manifest, device)
    torch.use_deterministic_algorithms(True)
    metrics, _, _ = evaluate_model(
        model,
        partition,
        vocabulary=corpus.vocabulary,
        device=device,
        deterministic=True,
    )
    invariance = model_order_invariance(
        model,
        partition,
        vocabulary=corpus.vocabulary,
        device=device,
        seed=int(manifest["seed"]),
    )
    print(
        json.dumps(
            {
                "format": "spider-v0.8-sre-evaluation-v1",
                "partition": args.partition,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "portable_weights_sha256": manifest["portable_weights_sha256"],
                "metrics": metrics,
                "row_permutation": invariance,
                "sealed_access_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
