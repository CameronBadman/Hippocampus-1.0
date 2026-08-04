#!/usr/bin/env python3
"""Run the frozen Spider v0.8 finalist on three public SRE demo cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hippocampus.sre import load_development_corpus, pack_partition  # noqa: E402
from evaluate_spider_v0_8_sre import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    load_selected_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "artifacts/spider_v0_8/DEMO_REPORT.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    corpus = load_development_corpus(cache_root=args.output_root / "cache")
    model, selected = load_selected_model(args.manifest, device)
    torch.use_deterministic_algorithms(True)
    model.eval()
    with torch.inference_mode():
        packed = pack_partition(
            corpus.demo,
            vocabulary=corpus.vocabulary,
            device=device,
            execution_policy="deterministic",
            validate=True,
        )
        output = model(packed)
    cases = []
    for case_index, case in enumerate(corpus.demo.cases):
        scores = output.scores[case_index].float().cpu()
        null = float(output.null_scores[case_index].item())
        order = torch.argsort(scores, descending=True, stable=True)
        selected_indices = torch.nonzero(scores > null).flatten().tolist()
        cases.append(
            {
                "case_id": case.case_id,
                "query": case.query_text,
                "incoming_observation": case.incoming_text,
                "null_score": null,
                "predicted_evidence": [
                    {
                        "candidate_id": case.candidates[index].candidate_id,
                        "score": float(scores[index]),
                        "status": case.candidates[index].status,
                        "memory_type": case.candidates[index].memory_type,
                        "text": case.candidates[index].text,
                        "required_by_public_demo": case.labels[index].relevant,
                    }
                    for index in selected_indices
                ],
                "top_five": [
                    {
                        "candidate_id": case.candidates[int(index)].candidate_id,
                        "score": float(scores[int(index)]),
                        "required_by_public_demo": case.labels[int(index)].relevant,
                        "text": case.candidates[int(index)].text,
                    }
                    for index in order[:5]
                ],
                "predicted_ids": sorted(
                    case.candidates[index].candidate_id for index in selected_indices
                ),
                "required_ids": sorted(case.positive_candidate_ids),
                "exact_public_demo_match": {
                    case.candidates[index].candidate_id for index in selected_indices
                }
                == case.positive_candidate_ids,
            }
        )
    report = {
        "format": "spider-v0.8-sre-public-demo-v1",
        "diagnostic_only": True,
        "aggregate_metric": None,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "portable_weights_sha256": selected["portable_weights_sha256"],
        "sealed_access_count": 0,
        "cases": cases,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for case in cases:
        print(f"\n{case['query']}")
        print(
            f"selected {len(case['predicted_ids'])} memories; "
            f"public-demo exact match={case['exact_public_demo_match']}"
        )
        for evidence in case["predicted_evidence"]:
            print(f"  {evidence['score']:+.3f}  {evidence['text']}")


if __name__ == "__main__":
    main()
