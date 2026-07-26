#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from hippocampus import PackConfig
from hippocampus.programs import (
    SyntheticManifoldRenderer,
    default_split_specs,
    generate_split_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    build_model,
    load_experiment,
    make_tiny_cases,
    parameter_count,
    train_oracle_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Spider v0 model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/spider_v0/tiny_overfit.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cases", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.config)
    dtype_names = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if args.device is not None or args.dtype is not None:
        device = (
            experiment.device
            if args.device is None
            else torch.device(args.device)
        )
        dtype = (
            experiment.dtype
            if args.dtype is None
            else dtype_names[args.dtype]
        )
        experiment = replace(
            experiment,
            device=device,
            dtype=dtype,
            pack_config=PackConfig(device=device, value_dtype=dtype),
        )
    loop = (
        experiment.training_config
        if args.steps is None
        else replace(experiment.training_config, steps=args.steps)
    )
    case_count = args.cases or int(
        experiment.raw.get("dataset", {}).get("cases", 64)
    )
    dataset_split = str(
        experiment.raw.get("dataset", {}).get("split", "train")
    )
    if dataset_split == "tiny_overfit":
        cases = make_tiny_cases(
            case_count=case_count,
            seed=int(experiment.raw["dataset"]["seed"]),
        )
    else:
        specs = {spec.name: spec for spec in default_split_specs()}
        cases = generate_split_cases(specs[dataset_split], limit=case_count)

    torch.manual_seed(loop.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(loop.seed)
    renderer = SyntheticManifoldRenderer(
        experiment.schema,
        query_dim=experiment.query_dim,
        seed=71,
    )
    pack_config = PackConfig(
        device=experiment.device,
        value_dtype=experiment.dtype,
    )
    batches = tuple(
        pack_rendered_cases(
            (case,),
            (
                renderer.render(
                    case,
                    row_permutation_seed=loop.seed + index,
                ),
            ),
            schema=experiment.schema,
            pack_config=pack_config,
        )
        for index, case in enumerate(cases)
    )
    model = build_model(experiment)
    output_dir = args.output_dir or (
        Path("artifacts/spider_v0/runs") / experiment.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    result = train_oracle_batches(
        model,
        batches,
        loop_config=loop,
        loss_config=experiment.loss_config,
        controller_config=experiment.controller_config,
        checkpoint_path=checkpoint,
    )
    metrics = {
        "status": "completed",
        "experiment": experiment.name,
        "config": str(args.config),
        "model_kind": experiment.raw["model"]["kind"],
        "parameter_count": parameter_count(model),
        "case_count": len(cases),
        "device": str(experiment.device),
        "dtype": str(experiment.dtype),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(experiment.device)
            if experiment.device.type == "cuda"
            else None
        ),
        "runtime_seconds": result.runtime_seconds,
        "initial": result.initial_metrics.as_dict(),
        "final": result.final_metrics.as_dict(),
        "checkpoint": str(checkpoint),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "history.jsonl").open("w") as handle:
        for record in result.records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
