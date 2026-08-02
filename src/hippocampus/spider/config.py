from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EdgeMode = Literal["standard", "compositional"]
TerminationMode = Literal["flat", "hierarchical", "factorized"]
ExpansionPolicy = Literal["threshold", "learned_null"]
EvidenceReadoutMode = Literal[
    "shared",
    "dedicated_pooled",
    "slot_aware",
]


@dataclass(frozen=True, slots=True)
class SpiderModelConfig:
    summary_dim: int
    context_dim: int
    edge_dim: int
    query_dim: int
    d_model: int = 128
    num_heads: int = 4
    num_blocks: int = 2
    path_rows: int = 8
    evidence_rows: int = 8
    edge_mode: EdgeMode = "standard"
    edge_transforms: int = 4
    adapter_rank: int = 16
    dropout: float = 0.0
    use_global_evidence: bool = True
    tied_recurrence: bool = True
    untied_rounds: int = 8
    control_width: int = 6
    termination_mode: TerminationMode = "flat"
    use_null_expansion: bool = False
    evidence_readout: EvidenceReadoutMode = "shared"

    def __post_init__(self) -> None:
        for name in (
            "summary_dim",
            "context_dim",
            "edge_dim",
            "query_dim",
            "d_model",
            "num_heads",
            "num_blocks",
            "path_rows",
            "evidence_rows",
            "untied_rounds",
            "control_width",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.edge_mode not in {"standard", "compositional"}:
            raise ValueError("edge_mode must be standard or compositional")
        if self.edge_mode == "compositional":
            if self.edge_transforms <= 0:
                raise ValueError("compositional mode requires edge transforms")
            if self.adapter_rank <= 0:
                raise ValueError("compositional mode requires a positive adapter rank")
        if self.termination_mode not in {
            "flat",
            "hierarchical",
            "factorized",
        }:
            raise ValueError(
                "termination_mode must be flat, hierarchical, or factorized"
            )
        if self.evidence_readout not in {
            "shared",
            "dedicated_pooled",
            "slot_aware",
        }:
            raise ValueError(
                "evidence_readout must be shared, dedicated_pooled, or "
                "slot_aware"
            )


@dataclass(frozen=True, slots=True)
class SparseControllerConfig:
    max_rounds: int = 8
    frontier_width: int = 32
    hypotheses_per_node: int = 2
    context_read_budget: int = 4
    search_budget: int = 4096
    max_depth: int = 12
    expand_threshold: float = 0.5
    context_threshold: float = 0.5
    evidence_threshold: float = 0.5
    evidence_selection_budget: int = 32
    expansion_policy: ExpansionPolicy = "threshold"

    def __post_init__(self) -> None:
        for name in (
            "max_rounds",
            "frontier_width",
            "hypotheses_per_node",
            "search_budget",
            "max_depth",
        ):
            if getattr(self, name) <= 0 and not (
                name == "search_budget" and getattr(self, name) == 0
            ):
                raise ValueError(f"{name} must be positive")
        if self.context_read_budget < 0:
            raise ValueError("context_read_budget must be non-negative")
        if self.evidence_selection_budget < 0:
            raise ValueError("evidence_selection_budget must be non-negative")
        for name in (
            "expand_threshold",
            "context_threshold",
            "evidence_threshold",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.expansion_policy not in {"threshold", "learned_null"}:
            raise ValueError(
                "expansion_policy must be threshold or learned_null"
            )
