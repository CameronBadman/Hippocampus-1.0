#!/usr/bin/env python3
"""Evaluate a frozen Spider v0.8 checkpoint on a non-sealed dev partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


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
DEFAULT_RESULT = DEFAULT_OUTPUT / "runs/V08S-T1-s1701/result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--partition",
        choices=("selection", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_selected_model(result_path: Path, device: torch.device):
    result = json.loads(result_path.read_text())
    if result["sealed_access_count"] != 0:
        raise RuntimeError("refusing a checkpoint associated with sealed access")
    config = json.loads((ROOT / "configs/spider_v0_8/T1.json").read_text())
    if result["arm"] != config["arm"]:
        raise ValueError("result and selected model configuration disagree")
    model = PackedSRECanonicalRetriever(SREModelConfig(**config["model"])).to(device)
    checkpoint = torch.load(
        result["selected_checkpoint"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    return model, result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    corpus = load_development_corpus(cache_root=args.output_root / "cache")
    partition = getattr(corpus, args.partition)
    model, result = load_selected_model(args.result, device)
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
        seed=int(result["seed"]),
    )
    print(
        json.dumps(
            {
                "format": "spider-v0.8-sre-evaluation-v1",
                "partition": args.partition,
                "checkpoint_sha256": result["selected_checkpoint_sha256"],
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
