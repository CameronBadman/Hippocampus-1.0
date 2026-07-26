#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import time
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
    evaluate_batches,
    load_experiment,
    parameter_count,
    train_oracle_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mechanical evaluator for one Spider AutoResearch trial."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--train-cases", type=int, default=64)
    parser.add_argument("--eval-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    return parser.parse_args()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _split_hash() -> str:
    index = json.loads(
        Path("artifacts/spider_v0/splits/MANIFEST_INDEX.json").read_text()
    )
    return str(index["aggregate_sha256"])


def _pack_cases(experiment, cases, renderer, row_offset: int):
    return tuple(
        pack_rendered_cases(
            (case,),
            (
                renderer.render(
                    case,
                    row_permutation_seed=(
                        experiment.training_config.seed + row_offset + index
                    ),
                ),
            ),
            schema=experiment.schema,
            pack_config=experiment.pack_config,
        )
        for index, case in enumerate(cases)
    )


def _primary_score(reports: dict[str, dict[str, object]]) -> float:
    weights = {
        "validation_id": 0.20,
        "validation_graph_size_ood": 0.20,
        "validation_path_length_ood": 0.20,
        "validation_topology_ood": 0.10,
        "validation_cardinality_ood": 0.10,
        "validation_equivalent_view_ood": 0.10,
        "validation_composition_ood": 0.10,
    }
    score = 0.0
    for split, weight in weights.items():
        rollout = reports[split]["rollout"]
        assert isinstance(rollout, dict)
        score += weight * float(rollout["termination_accuracy"])
    id_teacher = reports["validation_id"]["teacher_forced"]
    id_efficiency = reports["validation_id"]["efficiency"]
    assert isinstance(id_teacher, dict)
    assert isinstance(id_efficiency, dict)
    score -= 0.05 * float(id_teacher["invalid_expansion_rate"])
    score -= 0.02 * min(
        1.0,
        float(id_efficiency["mean_contexts_read"]) / 4.0,
    )
    return score


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    experiment = load_experiment(args.config)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device(args.device)
    experiment = replace(
        experiment,
        device=device,
        dtype=dtype,
        pack_config=PackConfig(device=device, value_dtype=dtype),
        training_config=replace(
            experiment.training_config,
            steps=args.steps,
            seed=args.seed,
            batch_size=min(8, experiment.training_config.batch_size),
            log_every=max(1, args.steps // 4),
        ),
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    renderer = SyntheticManifoldRenderer(
        experiment.schema,
        query_dim=experiment.query_dim,
        seed=71,
    )
    specs = {spec.name: spec for spec in default_split_specs()}
    train_cases = generate_split_cases(
        specs["train"],
        limit=args.train_cases,
    )
    train_batches = _pack_cases(experiment, train_cases, renderer, 0)
    model = build_model(experiment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    training = train_oracle_batches(
        model,
        train_batches,
        loop_config=experiment.training_config,
        loss_config=experiment.loss_config,
        controller_config=experiment.controller_config,
        checkpoint_path=checkpoint,
    )

    reports: dict[str, dict[str, object]] = {}
    row_mismatches = 0
    replay_mismatches = 0
    for split_name, spec in specs.items():
        if spec.sealed or split_name == "train":
            continue
        cases = generate_split_cases(spec, limit=args.eval_cases)
        batches = _pack_cases(experiment, cases, renderer, 10_000)
        permuted = _pack_cases(experiment, cases, renderer, 20_000)
        report = evaluate_batches(
            model,
            batches,
            split=split_name,
            controller_config=experiment.controller_config,
            permuted_batches=permuted,
            invariance_sample_limit=min(4, args.eval_cases),
        )
        reports[split_name] = report.as_dict()
        row_mismatches += int(
            report.invariance["row_permutation_decision_mismatches"]
        )
        replay_mismatches += int(
            report.invariance["deterministic_replay_mismatches"]
        )

    score = _primary_score(reports)
    finite = bool(torch.isfinite(torch.tensor(score)).item())
    status = (
        "accepted"
        if finite and row_mismatches == 0 and replay_mismatches == 0
        else "invalid"
    )
    runtime = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "status": status,
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_commit": _source_commit(),
        "dataset_split_hash": _split_hash(),
        "config_path": str(args.config),
        "config": experiment.raw,
        "parameter_count": parameter_count(model),
        "training_steps": args.steps,
        "train_case_count": args.train_cases,
        "evaluation_case_count_per_split": args.eval_cases,
        "seed": args.seed,
        "score": score,
        "metrics": reports,
        "training": {
            "initial": training.initial_metrics.as_dict(),
            "final": training.final_metrics.as_dict(),
            "runtime_seconds": training.runtime_seconds,
            "history": [asdict(record) for record in training.records],
        },
        "runtime_seconds": runtime,
        "peak_memory_bytes": peak_memory,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "row_permutation_mismatches": row_mismatches,
        "deterministic_replay_mismatches": replay_mismatches,
        "sealed_test_opened": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "failure_reason": (
            None
            if status == "accepted"
            else "non-finite score or invariance/determinism guard failure"
        ),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
