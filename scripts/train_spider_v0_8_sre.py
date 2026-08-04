#!/usr/bin/env python3
"""Train or evaluate one registered Spider v0.8 SRE arm."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hippocampus.sre import (  # noqa: E402
    PackedSRECanonicalRetriever,
    SREModelConfig,
    SRERetrievalLossConfig,
    evaluate_frozen_baseline,
    evaluate_model,
    load_development_corpus,
    model_order_invariance,
    pack_partition,
    seed_everything,
    sre_retrieval_loss,
)
from hippocampus.sre.experiment import load_json, save_checkpoint  # noqa: E402


DEFAULT_CACHE = ROOT / "artifacts/spider_v0_8/local_rtx5070ti/cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkpoint_key(metrics: dict[str, Any], step: int) -> tuple[float, ...]:
    components = metrics["components"]
    set_metrics = metrics["set_selection"]
    return (
        float(metrics["score"]),
        float(components["recall_at_8"]),
        float(components["macro_average_precision"]),
        -float(set_metrics["false_positives_per_case"]),
        -step,
    )


def _model_result(
    *,
    model: PackedSRECanonicalRetriever,
    corpus,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    selected_step: int,
    checkpoint_path: Path,
    started: float,
    peak_memory: int,
) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    batch_size = int(config["evaluation_batch_size"])
    evaluation, scores, null_scores = evaluate_model(
        model,
        corpus.evaluation,
        vocabulary=corpus.vocabulary,
        device=device,
        batch_size=batch_size,
        deterministic=True,
    )
    replay, replay_scores, replay_null = evaluate_model(
        model,
        corpus.evaluation,
        vocabulary=corpus.vocabulary,
        device=device,
        batch_size=batch_size,
        deterministic=True,
    )
    score_mismatches = int(np.count_nonzero(scores != replay_scores))
    null_mismatches = int(np.count_nonzero(null_scores != replay_null))
    decision_mismatches = int(
        np.count_nonzero(
            (scores > null_scores[:, None])
            != (replay_scores > replay_null[:, None])
        )
    )
    invariance = model_order_invariance(
        model,
        corpus.evaluation,
        vocabulary=corpus.vocabulary,
        device=device,
        seed=seed,
    )
    return {
        "format": "spider-v0.8-sre-result-v1",
        "arm": config["arm"],
        "seed": seed,
        "source_commit": _source_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_step": selected_step,
        "selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_sha256": _sha256(checkpoint_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "evaluation": evaluation,
        "deterministic_replay": {
            "score_mismatch_count": score_mismatches,
            "null_mismatch_count": null_mismatches,
            "decision_mismatch_count": decision_mismatches,
            "metric_score_delta": abs(evaluation["score"] - replay["score"]),
        },
        "row_permutation": invariance,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak_memory,
        "sealed_access_count": 0,
    }


def _run_frozen(config: dict[str, Any], args: argparse.Namespace, corpus) -> dict[str, Any]:
    metrics, scores, null_scores = evaluate_frozen_baseline(
        corpus.evaluation,
        vocabulary=corpus.vocabulary,
        device=args.device,
    )
    replay, replay_scores, replay_null = evaluate_frozen_baseline(
        corpus.evaluation,
        vocabulary=corpus.vocabulary,
        device=args.device,
    )
    return {
        "format": "spider-v0.8-sre-result-v1",
        "arm": config["arm"],
        "seed": args.seed,
        "source_commit": _source_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_step": 0,
        "selected_checkpoint": None,
        "selected_checkpoint_sha256": None,
        "parameter_count": 0,
        "evaluation": metrics,
        "deterministic_replay": {
            "score_mismatch_count": int(np.count_nonzero(scores != replay_scores)),
            "null_mismatch_count": int(np.count_nonzero(null_scores != replay_null)),
            "decision_mismatch_count": int(
                np.count_nonzero(
                    (scores > null_scores[:, None])
                    != (replay_scores > replay_null[:, None])
                )
            ),
            "metric_score_delta": abs(metrics["score"] - replay["score"]),
        },
        "row_permutation": {
            "maximum_score_delta": 0.0,
            "decision_mismatch_count": 0,
            "null_score_delta": 0.0,
            "case_count": 100,
        },
        "runtime_seconds": 0.0,
        "peak_cuda_memory_bytes": 0,
        "sealed_access_count": 0,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    corpus = load_development_corpus(cache_root=args.cache_root)
    if not config.get("trainable", False):
        result = _run_frozen(config, args, corpus)
        _write(args.output_dir / "result.json", result)
        print(json.dumps(result, sort_keys=True))
        return

    steps = int(args.steps or config["steps"])
    interval = int(config["checkpoint_interval"])
    if steps <= 0 or interval <= 0:
        raise ValueError("training steps and checkpoint interval must be positive")
    model = PackedSRECanonicalRetriever(SREModelConfig(**config["model"])).to(device)
    loss_config = SRERetrievalLossConfig(**config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(args.seed + 1)
    batch_size = int(config["batch_size"])
    permutation = torch.randperm(len(corpus.train.cases), generator=generator)
    offset = 0
    best_key: tuple[float, ...] | None = None
    best_state = None
    best_step = 0
    checkpoints = []
    loss_log = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, steps + 1):
        if offset + batch_size > len(permutation):
            permutation = torch.randperm(len(corpus.train.cases), generator=generator)
            offset = 0
        indices = permutation[offset : offset + batch_size]
        offset += batch_size
        packed = pack_partition(
            corpus.train.select(indices),
            vocabulary=corpus.vocabulary,
            device=device,
        )
        model.train()
        output = model(packed)
        loss, components = sre_retrieval_loss(output, packed, loss_config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["gradient_clip"]),
        )
        optimizer.step()
        if step == 1 or step % 25 == 0:
            loss_log.append(
                {
                    "step": step,
                    **{
                        name: float(value.detach().item())
                        for name, value in components.items()
                    },
                    "gradient_norm": float(gradient_norm),
                }
            )
        if step % interval == 0 or step == steps:
            selection, _, _ = evaluate_model(
                model,
                corpus.selection,
                vocabulary=corpus.vocabulary,
                device=device,
                batch_size=int(config["evaluation_batch_size"]),
            )
            checkpoint = args.output_dir / f"checkpoint_step_{step:06d}.pt"
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
                seed=args.seed,
                selection_metrics=selection,
            )
            key = _checkpoint_key(selection, step)
            checkpoints.append(
                {"step": step, "selection": selection, "checkpoint": str(checkpoint)}
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "selection_score": selection["score"],
                        "components": selection["components"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise AssertionError("training produced no selectable checkpoint")
    model.load_state_dict(best_state)
    selected_checkpoint = args.output_dir / f"checkpoint_step_{best_step:06d}.pt"
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    result = _model_result(
        model=model,
        corpus=corpus,
        config=config,
        device=device,
        seed=args.seed,
        selected_step=best_step,
        checkpoint_path=selected_checkpoint,
        started=started,
        peak_memory=peak_memory,
    )
    result["selection_history"] = checkpoints
    result["loss_history"] = loss_log
    _write(args.output_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
