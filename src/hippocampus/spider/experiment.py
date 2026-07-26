from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..config import GraphSchema, PackConfig
from .baselines import FlatTransformerScorer, PooledScorer
from .config import SparseControllerConfig, SpiderModelConfig
from .losses import SpiderLossConfig
from .model import CandidateScorerBase, SpiderModel
from .controller import ActionSchedule
from .training import TrainingLoopConfig


@dataclass(frozen=True, slots=True)
class ResolvedExperiment:
    raw: dict[str, Any]
    schema: GraphSchema
    query_dim: int
    model_config: SpiderModelConfig
    controller_config: SparseControllerConfig
    training_config: TrainingLoopConfig
    loss_config: SpiderLossConfig
    device: torch.device
    dtype: torch.dtype
    pack_config: PackConfig

    @property
    def name(self) -> str:
        return str(self.raw["name"])


def _resolve_device(value: str | None) -> torch.device:
    if value in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _resolve_dtype(value: str | None) -> torch.dtype:
    mapping = {
        None: torch.float32,
        "auto": torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unsupported training dtype {value!r}") from exc


def load_experiment(path: str | Path) -> ResolvedExperiment:
    config_path = Path(path)
    raw = json.loads(config_path.read_text())
    schema_data = raw["schema"]
    schema = GraphSchema(
        summary_dim=int(schema_data["summary_dim"]),
        context_dim=int(schema_data["context_dim"]),
        edge_dim=int(schema_data["edge_dim"]),
    )
    query_dim = int(schema_data["query_dim"])
    model_data = raw["model"]
    model_config = SpiderModelConfig(
        summary_dim=schema.summary_dim,
        context_dim=schema.context_dim,
        edge_dim=schema.edge_dim,
        query_dim=query_dim,
        d_model=int(model_data["d_model"]),
        num_heads=int(model_data["num_heads"]),
        num_blocks=max(1, int(model_data["num_blocks"])),
        path_rows=int(model_data["path_rows"]),
        evidence_rows=int(model_data["evidence_rows"]),
        edge_mode=str(model_data.get("edge_mode", "standard")),
        edge_transforms=max(1, int(model_data.get("edge_transforms", 1))),
        adapter_rank=max(1, int(model_data.get("adapter_rank", 1))),
        dropout=float(model_data.get("dropout", 0.0)),
        use_global_evidence=bool(model_data.get("use_global_evidence", True)),
        tied_recurrence=bool(model_data.get("tied_recurrence", True)),
        untied_rounds=int(model_data.get("untied_rounds", 8)),
        termination_mode=str(model_data.get("termination_mode", "flat")),
    )
    controller_data = raw["controller"]
    controller_config = SparseControllerConfig(
        max_rounds=int(controller_data["max_rounds"]),
        frontier_width=int(controller_data["frontier_width"]),
        hypotheses_per_node=int(controller_data["hypotheses_per_node"]),
        context_read_budget=int(controller_data["context_read_budget"]),
        search_budget=int(controller_data.get("search_budget", 4096)),
        max_depth=int(controller_data["max_depth"]),
        expand_threshold=float(
            controller_data.get("expand_threshold", 0.5)
        ),
        context_threshold=float(
            controller_data.get("context_threshold", 0.5)
        ),
        evidence_threshold=float(
            controller_data.get("evidence_threshold", 0.5)
        ),
        evidence_selection_budget=int(
            controller_data.get("evidence_selection_budget", 32)
        ),
    )
    training_data = raw["training"]
    training_config = TrainingLoopConfig(
        steps=int(training_data["steps"]),
        batch_size=int(training_data["batch_size"]),
        learning_rate=float(training_data["learning_rate"]),
        weight_decay=float(training_data.get("weight_decay", 0.0)),
        seed=int(training_data["seed"]),
        log_every=int(training_data.get("log_every", 25)),
        max_grad_norm=float(training_data.get("max_grad_norm", 5.0)),
        oracle_fraction_schedule=tuple(
            float(value)
            for value in training_data.get(
                "oracle_fraction_schedule",
                (1.0,),
            )
        ),
        action_schedule=tuple(
            ActionSchedule(
                frontier=float(item["frontier"]),
                context=float(item["context"]),
                evidence=float(item["evidence"]),
                termination=float(item["termination"]),
            )
            for item in training_data.get("action_schedule", ())
        ),
    )
    loss_config = SpiderLossConfig(**raw["loss"])
    device = _resolve_device(training_data.get("device"))
    dtype = _resolve_dtype(training_data.get("dtype"))
    pack_config = PackConfig(device=device, value_dtype=dtype)
    return ResolvedExperiment(
        raw=raw,
        schema=schema,
        query_dim=query_dim,
        model_config=model_config,
        controller_config=controller_config,
        training_config=training_config,
        loss_config=loss_config,
        device=device,
        dtype=dtype,
        pack_config=pack_config,
    )


def build_model(experiment: ResolvedExperiment) -> CandidateScorerBase:
    kind = str(experiment.raw["model"]["kind"])
    model_types: dict[str, type[CandidateScorerBase]] = {
        "pooled": PooledScorer,
        "flat_transformer": FlatTransformerScorer,
        "spider": SpiderModel,
    }
    try:
        model_type = model_types[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported model kind {kind!r}") from exc
    return model_type(experiment.model_config).to(
        device=experiment.device,
        dtype=experiment.dtype,
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
