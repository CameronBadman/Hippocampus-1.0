#!/usr/bin/env python3
"""Audit which runtime inputs the frozen Spider v0.8 finalist uses."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hippocampus.sre import (  # noqa: E402
    PackedSRECanonicalRetriever,
    SREModelConfig,
    SREPartition,
    evaluate_model,
    load_development_corpus,
)


DEFAULT_OUTPUT = ROOT / "artifacts/spider_v0_8/local_rtx5070ti"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _partition_with(partition: SREPartition, **changes) -> SREPartition:
    return SREPartition(
        partition.cases,
        replace(partition.encoded, **changes),
    )


def _shuffle_rows(values: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    permutations = torch.stack(
        [torch.randperm(values.shape[1], generator=generator) for _ in values]
    )
    gather = permutations[:, :, None].expand_as(values)
    return torch.gather(values, 1, gather)


def _ablations(partition: SREPartition, seed: int) -> dict[str, SREPartition]:
    encoded = partition.encoded
    case_order = torch.roll(torch.arange(encoded.case_count), shifts=1)
    return {
        "full": partition,
        "no_semantic_embeddings": _partition_with(
            partition,
            query_embeddings=torch.zeros_like(encoded.query_embeddings),
            incoming_embeddings=torch.zeros_like(encoded.incoming_embeddings),
            candidate_embeddings=torch.zeros_like(encoded.candidate_embeddings),
        ),
        "no_runtime_features": _partition_with(
            partition,
            candidate_features=torch.zeros_like(encoded.candidate_features),
        ),
        "shuffled_candidate_semantics": _partition_with(
            partition,
            candidate_embeddings=_shuffle_rows(
                encoded.candidate_embeddings,
                seed,
            ),
        ),
        "shuffled_runtime_features": _partition_with(
            partition,
            candidate_features=_shuffle_rows(
                encoded.candidate_features,
                seed + 1,
            ),
        ),
        "wrong_case_query": _partition_with(
            partition,
            query_embeddings=encoded.query_embeddings[case_order],
            incoming_embeddings=encoded.incoming_embeddings[case_order],
            incoming_present=encoded.incoming_present[case_order],
        ),
    }


def _load_model(result: dict, device: torch.device):
    config = json.loads((ROOT / "configs/spider_v0_8/T1.json").read_text())
    model = PackedSRECanonicalRetriever(SREModelConfig(**config["model"])).to(device)
    checkpoint = torch.load(
        result["selected_checkpoint"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    corpus = load_development_corpus(cache_root=args.output_root / "cache")
    torch.use_deterministic_algorithms(True)
    run_paths = sorted((args.output_root / "runs").glob("V08*-T1-*/result.json"))
    if len(run_paths) != 3:
        raise RuntimeError("the T1 three-seed confirmation must exist before ablation")
    records = []
    for path in run_paths:
        result = json.loads(path.read_text())
        seed = int(result["seed"])
        model = _load_model(result, device)
        for name, partition in _ablations(corpus.evaluation, seed).items():
            metrics, _, _ = evaluate_model(
                model,
                partition,
                vocabulary=corpus.vocabulary,
                device=device,
                batch_size=16,
                deterministic=True,
            )
            records.append(
                {
                    "seed": seed,
                    "ablation": name,
                    "score": metrics["score"],
                    "components": metrics["components"],
                    "set_selection": metrics["set_selection"],
                    "sealed_access_count": 0,
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "ablation": name,
                        "score": metrics["score"],
                        "exact_set": metrics["set_selection"]["exact_set_accuracy"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    names = sorted({record["ablation"] for record in records})
    aggregate = {
        name: {
            "mean_score": statistics.mean(
                record["score"] for record in records if record["ablation"] == name
            ),
            "mean_exact_set_accuracy": statistics.mean(
                record["set_selection"]["exact_set_accuracy"]
                for record in records
                if record["ablation"] == name
            ),
            "mean_precision": statistics.mean(
                record["set_selection"]["precision"]
                for record in records
                if record["ablation"] == name
            ),
            "mean_recall": statistics.mean(
                record["set_selection"]["recall"]
                for record in records
                if record["ablation"] == name
            ),
        }
        for name in names
    }
    output = {
        "format": "spider-v0.8-sre-input-ablation-v1",
        "diagnostic_only": True,
        "sealed_access_count": 0,
        "records": records,
        "aggregate": aggregate,
    }
    target = args.output_root / "INPUT_ABLATIONS.json"
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
