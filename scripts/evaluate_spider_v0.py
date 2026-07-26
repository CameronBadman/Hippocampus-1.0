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
    pack_rendered_cases,
)
from hippocampus.spider import build_model, evaluate_batches, load_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Spider v0.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="validation_id")
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-sealed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = {spec.name: spec for spec in default_split_specs()}
    try:
        spec = specs[args.split]
    except KeyError as exc:
        raise SystemExit(f"unknown split {args.split!r}") from exc
    if spec.sealed and not args.allow_sealed:
        raise SystemExit(
            "sealed test access denied; freeze the finalist and pass --allow-sealed once"
        )

    experiment = load_experiment(args.config)
    model = build_model(experiment)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=experiment.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    cases = generate_split_cases(spec, limit=args.cases)
    renderer = SyntheticManifoldRenderer(
        experiment.schema,
        query_dim=experiment.query_dim,
        seed=71,
    )

    def pack(row_seed_offset: int):
        return tuple(
            pack_rendered_cases(
                (case,),
                (
                    renderer.render(
                        case,
                        row_permutation_seed=(
                            experiment.training_config.seed
                            + row_seed_offset
                            + index
                        ),
                    ),
                ),
                schema=experiment.schema,
                pack_config=experiment.pack_config,
            )
            for index, case in enumerate(cases)
        )

    batches = pack(0)
    permuted = pack(100_000)
    report = evaluate_batches(
        model,
        batches,
        split=spec.name,
        controller_config=experiment.controller_config,
        permuted_batches=permuted,
    )
    payload = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "sealed_test_opened": spec.sealed,
        **report.as_dict(),
    }
    output = args.output or (
        Path("artifacts/spider_v0/evaluations")
        / f"{experiment.name}-{spec.name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
