from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import torch

from ..programs.batching import PackedProgramBatch
from .hypothesis import HypothesisBatch
from .model import CandidateScorerBase


class HorizonMode(str, Enum):
    """When learned termination is allowed to end controller execution."""

    LEARNED = "learned"
    FIXED = "fixed"
    ORACLE_REQUIRED = "oracle_required"


class PathStateIntervention(str, Enum):
    """Diagnostic replacements for state carried between controller rounds."""

    NONE = "none"
    RESET = "reset"
    DETACH = "detach"
    SHUFFLE = "shuffle"
    POOLED_CURRENT_NODE = "pooled_current_node"


@dataclass(frozen=True, slots=True)
class ControllerExecutionPolicy:
    """Controller execution semantics independent of model architecture.

    Oracle-required horizons are supervisor-side diagnostics. The resolved
    horizon is never included in model-visible controller features.
    """

    horizon_mode: HorizonMode = HorizonMode.LEARNED
    fixed_rounds: int | None = None
    path_state_intervention: PathStateIntervention = PathStateIntervention.NONE
    intervention_seed: int = 0

    def __post_init__(self) -> None:
        if self.horizon_mode is HorizonMode.FIXED:
            if self.fixed_rounds is None or self.fixed_rounds <= 0:
                raise ValueError("fixed execution requires a positive horizon")
        elif self.fixed_rounds is not None:
            raise ValueError("fixed_rounds is valid only for fixed execution")

    @classmethod
    def learned(
        cls,
        *,
        intervention: PathStateIntervention = PathStateIntervention.NONE,
        seed: int = 0,
    ) -> "ControllerExecutionPolicy":
        return cls(
            horizon_mode=HorizonMode.LEARNED,
            path_state_intervention=intervention,
            intervention_seed=seed,
        )

    @classmethod
    def fixed(
        cls,
        rounds: int,
        *,
        intervention: PathStateIntervention = PathStateIntervention.NONE,
        seed: int = 0,
    ) -> "ControllerExecutionPolicy":
        return cls(
            horizon_mode=HorizonMode.FIXED,
            fixed_rounds=rounds,
            path_state_intervention=intervention,
            intervention_seed=seed,
        )

    @classmethod
    def oracle_required(
        cls,
        *,
        intervention: PathStateIntervention = PathStateIntervention.NONE,
        seed: int = 0,
    ) -> "ControllerExecutionPolicy":
        return cls(
            horizon_mode=HorizonMode.ORACLE_REQUIRED,
            path_state_intervention=intervention,
            intervention_seed=seed,
        )

    @property
    def suppresses_intermediate_termination(self) -> bool:
        return self.horizon_mode is not HorizonMode.LEARNED

    def resolve_round_limit(
        self,
        batch: PackedProgramBatch,
        *,
        configured_max_rounds: int,
    ) -> int:
        if self.horizon_mode is HorizonMode.LEARNED:
            return configured_max_rounds
        if self.horizon_mode is HorizonMode.FIXED:
            assert self.fixed_rounds is not None
            required = self.fixed_rounds
        else:
            required = max(len(case.trace.rounds) for case in batch.cases)
        if required > configured_max_rounds:
            raise ValueError(
                f"diagnostic horizon {required} exceeds configured max_rounds "
                f"{configured_max_rounds}"
            )
        return required


def _graph_local_shuffle(
    graph_ids: torch.Tensor,
    *,
    round_index: int,
    seed: int,
) -> torch.Tensor:
    """Return a seeded graph-local path-state source permutation."""

    count = int(graph_ids.numel())
    permutation = torch.arange(count, dtype=torch.int64)
    if count < 2:
        return permutation.to(graph_ids.device)
    graph_ids_cpu = graph_ids.detach().to(device="cpu", dtype=torch.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        (int(seed) + 1_000_003 * int(round_index)) % (2**63 - 1)
    )
    for graph_id in torch.unique(graph_ids_cpu, sorted=True).tolist():
        positions = torch.nonzero(
            graph_ids_cpu == graph_id,
            as_tuple=False,
        ).flatten()
        if positions.numel() > 1:
            local = torch.randperm(positions.numel(), generator=generator)
            permutation[positions] = positions[local]
    return permutation.to(graph_ids.device)


def apply_path_state_intervention(
    model: CandidateScorerBase,
    batch: PackedProgramBatch,
    hypotheses: HypothesisBatch,
    *,
    intervention: PathStateIntervention,
    round_index: int,
    seed: int,
) -> HypothesisBatch:
    """Apply one diagnostic intervention without changing controller metadata."""

    resolved = PathStateIntervention(intervention)
    if (
        resolved is PathStateIntervention.NONE
        or round_index == 0
        or hypotheses.count == 0
    ):
        return hypotheses
    if resolved is PathStateIntervention.RESET:
        path_state = model.initial_path_state(batch, hypotheses.graph_ids)
    elif resolved is PathStateIntervention.DETACH:
        path_state = hypotheses.path_state.detach()
    elif resolved is PathStateIntervention.SHUFFLE:
        permutation = _graph_local_shuffle(
            hypotheses.graph_ids,
            round_index=round_index,
            seed=seed,
        )
        path_state = hypotheses.path_state[permutation]
    else:
        path_state = model.pooled_current_node_path_state(batch, hypotheses)
    return replace(hypotheses, path_state=path_state).validate()

