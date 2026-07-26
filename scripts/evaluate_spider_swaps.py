#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hippocampus.programs import (
    SyntheticManifoldRenderer,
    default_split_specs,
    generate_split_cases,
    make_equivalent_view,
    pack_rendered_cases,
    swap_aligned_edge_manifolds,
)
from hippocampus.spider import (
    build_model,
    evaluate_oracle_batches,
    load_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-view functional edge-manifold swapping."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spider_v0/evaluations/functional-swaps.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.config)
    model = build_model(experiment)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=experiment.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    validation_id = next(
        spec
        for spec in default_split_specs()
        if spec.name == "validation_id"
    )
    cases = generate_split_cases(validation_id, limit=args.cases)
    renderer = SyntheticManifoldRenderer(
        experiment.schema,
        query_dim=experiment.query_dim,
        seed=71,
    )
    base_batches = []
    swapped_batches = []
    changed_edge_manifolds = 0
    total_edge_manifolds = 0
    for index, case in enumerate(cases):
        view = make_equivalent_view(case, seed=900_000 + index)
        base = renderer.render(case, row_permutation_seed=index)
        donor = renderer.render(view, row_permutation_seed=10_000 + index)
        swap = swap_aligned_edge_manifolds(case, base, view, donor)
        changed_edge_manifolds += sum(
            not torch.equal(original, changed)
            for original, changed in zip(
                base.edges,
                swap.rendered.edges,
                strict=True,
            )
        )
        total_edge_manifolds += len(base.edges)
        base_batches.append(
            pack_rendered_cases(
                (case,),
                (base,),
                schema=experiment.schema,
                pack_config=experiment.pack_config,
            )
        )
        swapped_batches.append(
            pack_rendered_cases(
                (case,),
                (swap.rendered,),
                schema=experiment.schema,
                pack_config=experiment.pack_config,
            )
        )
    base_loss, base_metrics = evaluate_oracle_batches(model, base_batches)
    swap_loss, swap_metrics = evaluate_oracle_batches(model, swapped_batches)
    report = {
        "case_count": len(cases),
        "checkpoint": str(args.checkpoint),
        "no_swap_training": {
            "status": "evaluated",
            "oracle_loss": base_loss,
            "metrics": base_metrics.as_dict(),
        },
        "swap_augmentation": {
            "status": "not_run",
            "reason": "outside the frozen 12-accepted-experiment budget",
        },
        "swap_consistency_loss": {
            "status": "implemented_not_trained",
            "reason": "outside the frozen 12-accepted-experiment budget",
        },
        "cross_view_swap_without_swap_training": {
            "status": "evaluated",
            "oracle_loss": swap_loss,
            "metrics": swap_metrics.as_dict(),
            "candidate_top1_delta": (
                swap_metrics.priority_top1 - base_metrics.priority_top1
            ),
            "termination_accuracy_delta": (
                swap_metrics.termination_accuracy
                - base_metrics.termination_accuracy
            ),
        },
        "changed_edge_manifold_fraction": (
            changed_edge_manifolds / max(1, total_edge_manifolds)
        ),
        "interpretation": (
            "Functional swapping is an experimental ablation; this report "
            "does not treat coordinate equality as a target."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
