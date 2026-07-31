from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import random
from typing import Protocol

import torch
from torch.nn import functional as F

from ..programs.batching import PackedProgramBatch
from ..programs.schema import TerminationDecision
from ..topology import FrontierExpansion
from .config import SparseControllerConfig
from .execution import (
    ControllerExecutionPolicy,
    HorizonMode,
    apply_path_state_intervention,
)
from .hypothesis import HypothesisBatch
from .model import CandidateScorerBase
from .terminator import TerminationOutput
from .types import CandidateOutputs


_TERMINATION_CLASSES = (
    TerminationDecision.CONTINUE,
    TerminationDecision.ANSWER,
    TerminationDecision.UNKNOWN_ABSENT,
    TerminationDecision.UNKNOWN_CONFLICT,
    TerminationDecision.UNKNOWN_INCOMPLETE,
    TerminationDecision.UNKNOWN_UNSUPPORTED,
)


class ActionSource(str, Enum):
    ORACLE = "oracle"
    MODEL = "model"


@dataclass(frozen=True, slots=True)
class ActionSchedule:
    """Independent teacher-forcing probabilities for discrete actions."""

    frontier: float
    context: float
    evidence: float
    termination: float

    def __post_init__(self) -> None:
        for name in ("frontier", "context", "evidence", "termination"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} teacher-forcing fraction must be in [0, 1]")

    @classmethod
    def oracle_only(cls) -> "ActionSchedule":
        return cls(1.0, 1.0, 1.0, 1.0)

    @classmethod
    def model_only(cls) -> "ActionSchedule":
        return cls(0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class TraceLedgerEntry:
    node_id: int
    edge_id: int
    arc_id: int
    round_index: int
    frontier_position: int
    parent_trace_id: int
    context_read: bool


@dataclass(frozen=True, slots=True)
class ContextLedgerEntry:
    node_id: int
    edge_id: int
    arc_id: int
    round_index: int
    frontier_position: int
    parent_trace_id: int


@dataclass(frozen=True, slots=True)
class EvidenceLedgerEntry:
    node_id: int
    edge_id: int
    arc_id: int
    round_index: int
    frontier_position: int
    parent_trace_id: int
    context_read: bool


@dataclass(frozen=True, slots=True)
class ControllerState:
    round_index: int
    arcs_scored: int
    contexts_read: int
    evidence_selected: int
    search_budget_exhausted: bool
    context_budget_exhausted: bool
    evidence_budget_exhausted: bool
    frontier_empty: bool
    last_expansion_had_arcs: bool
    last_expansion_truncated: bool
    deliberate_empty_frontier: bool
    depth_exhausted: bool
    trace_ledger: tuple[TraceLedgerEntry, ...]
    context_ledger: tuple[ContextLedgerEntry, ...]
    evidence_ledger: tuple[EvidenceLedgerEntry, ...]

    @classmethod
    def initial(cls) -> "ControllerState":
        return cls(
            round_index=0,
            arcs_scored=0,
            contexts_read=0,
            evidence_selected=0,
            search_budget_exhausted=False,
            context_budget_exhausted=False,
            evidence_budget_exhausted=False,
            frontier_empty=False,
            last_expansion_had_arcs=False,
            last_expansion_truncated=False,
            deliberate_empty_frontier=False,
            depth_exhausted=False,
            trace_ledger=(),
            context_ledger=(),
            evidence_ledger=(),
        )


@dataclass(frozen=True, slots=True)
class ControllerProposal:
    full_expansion: FrontierExpansion
    expansion: FrontierExpansion
    candidate_control_features: torch.Tensor
    candidate_outputs: CandidateOutputs
    depth_eligible: torch.Tensor
    search_limit: int
    context_limit: int
    evidence_limit: int
    full_arc_count: int
    search_truncated: bool
    candidate_graph_ids: torch.Tensor
    null_expansion_logits: torch.Tensor | None = None
    pre_context_outputs: CandidateOutputs | None = None
    context_refined: bool = False


@dataclass(frozen=True, slots=True)
class ControllerActions:
    frontier_candidate_indices: torch.Tensor
    context_candidate_indices: torch.Tensor
    evidence_candidate_indices: torch.Tensor
    frontier_source: ActionSource
    context_source: ActionSource
    evidence_source: ActionSource
    termination_source: ActionSource

    @classmethod
    def empty(
        cls,
        device: torch.device | str,
        *,
        source: ActionSource = ActionSource.MODEL,
    ) -> "ControllerActions":
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return cls(
            frontier_candidate_indices=empty,
            context_candidate_indices=empty,
            evidence_candidate_indices=empty,
            frontier_source=source,
            context_source=source,
            evidence_source=source,
            termination_source=source,
        )


@dataclass(frozen=True, slots=True)
class ControllerTransition:
    proposal: ControllerProposal
    actions: ControllerActions
    refined_outputs: CandidateOutputs
    next_hypotheses: HypothesisBatch
    next_evidence: torch.Tensor
    next_controller_state: ControllerState
    termination_control: torch.Tensor

    # Compatibility properties for the v0 ControllerStep surface.
    @property
    def expansion(self) -> FrontierExpansion:
        return self.proposal.expansion

    @property
    def outputs(self) -> CandidateOutputs:
        return self.refined_outputs

    @property
    def selected_candidate_indices(self) -> torch.Tensor:
        return self.actions.frontier_candidate_indices

    @property
    def context_candidate_indices(self) -> torch.Tensor:
        return self.actions.context_candidate_indices

    @property
    def evidence(self) -> torch.Tensor:
        return self.next_evidence

    @property
    def state(self) -> ControllerState:
        return self.next_controller_state


# Retain the public v0 name while making it the shared transition record.
ControllerStep = ControllerTransition


@dataclass(frozen=True, slots=True)
class ControllerActionSelection:
    """Context-refined proposal and the actions chosen from it."""

    proposal: ControllerProposal
    actions: ControllerActions


@dataclass(frozen=True, slots=True)
class ActionDiagnostic:
    round_index: int
    frontier_source: ActionSource
    context_source: ActionSource
    evidence_source: ActionSource
    termination_source: ActionSource
    frontier_candidate_indices: tuple[int, ...]
    context_candidate_indices: tuple[int, ...]
    evidence_candidate_indices: tuple[int, ...]
    executed_termination: TerminationDecision


@dataclass(frozen=True, slots=True)
class ControllerRoundRecord:
    """One exact model-policy round for evaluator-side observation."""

    hypotheses: HypothesisBatch
    controller_state: ControllerState
    proposal: ControllerProposal
    actions: ControllerActions
    transition: ControllerTransition
    termination_output: TerminationOutput | None
    termination: tuple[TerminationDecision, ...]


@dataclass(frozen=True, slots=True)
class ControllerResult:
    hypotheses: HypothesisBatch
    evidence: torch.Tensor
    termination: tuple[TerminationDecision, ...]
    selected_arc_trace: tuple[tuple[int, ...], ...]
    trace_ledger: tuple[TraceLedgerEntry, ...]
    context_ledger: tuple[ContextLedgerEntry, ...]
    evidence_ledger: tuple[EvidenceLedgerEntry, ...]
    action_diagnostics: tuple[ActionDiagnostic, ...]
    final_termination_logits: torch.Tensor
    rounds: int
    arcs_scored: int
    contexts_read: int
    round_records: tuple[ControllerRoundRecord, ...] = ()


class _CandidateTargets(Protocol):
    acceptable: torch.Tensor
    context_has_value: torch.Tensor
    include_as_evidence: torch.Tensor
    remaining_cost: torch.Tensor


class _StateSupervision(Protocol):
    candidates: _CandidateTargets


def _stable_priority_order(
    expansion: FrontierExpansion,
    priorities: torch.Tensor,
) -> torch.Tensor:
    if priorities.ndim != 1 or priorities.numel() != expansion.total_arcs:
        raise ValueError("priorities must align with expanded arcs")
    if priorities.numel() and not bool(torch.isfinite(priorities).all().item()):
        raise ValueError("candidate priorities must be finite")
    order = torch.arange(
        priorities.numel(),
        dtype=torch.int64,
        device=priorities.device,
    )
    # Stable least-to-most-significant sorts make priority primary, arc ID the
    # final snapshot-local tie-breaker, and frontier occurrence the last
    # deterministic fallback for duplicate arc occurrences.
    order = order[
        torch.argsort(
            expansion.frontier_positions[order],
            stable=True,
        )
    ]
    order = order[
        torch.argsort(
            expansion.arc_ids[order].to(torch.int64),
            stable=True,
        )
    ]
    order = order[
        torch.argsort(
            priorities[order],
            descending=True,
            stable=True,
        )
    ]
    return order


def stable_candidate_selection(
    expansion: FrontierExpansion,
    priorities: torch.Tensor,
    *,
    frontier_width: int,
    hypotheses_per_node: int,
) -> torch.Tensor:
    """Stable vectorised global ranking with a per-destination cap."""

    if frontier_width <= 0 or hypotheses_per_node <= 0:
        raise ValueError("selection limits must be positive")
    order = _stable_priority_order(expansion, priorities)
    if order.numel() == 0:
        return order

    destinations = expansion.destination_node_ids[order].to(torch.int64)
    group_permutation = torch.argsort(destinations, stable=True)
    grouped_candidates = order[group_permutation]
    grouped_destinations = destinations[group_permutation]
    starts_mask = torch.ones(
        grouped_destinations.numel(),
        dtype=torch.bool,
        device=priorities.device,
    )
    starts_mask[1:] = grouped_destinations[1:] != grouped_destinations[:-1]
    group_starts = torch.nonzero(starts_mask, as_tuple=False).flatten()
    group_ids = torch.cumsum(starts_mask.to(torch.int64), dim=0) - 1
    local_rank = (
        torch.arange(
            grouped_destinations.numel(),
            dtype=torch.int64,
            device=priorities.device,
        )
        - group_starts[group_ids]
    )
    kept = grouped_candidates[local_rank < hypotheses_per_node]
    score_rank = torch.empty_like(order)
    score_rank[order] = torch.arange(
        order.numel(),
        dtype=torch.int64,
        device=priorities.device,
    )
    kept = kept[torch.argsort(score_rank[kept], stable=True)]
    return kept[:frontier_width]


def _slice_expansion(
    expansion: FrontierExpansion,
    indices: torch.Tensor,
    *,
    frontier_count: int,
) -> FrontierExpansion:
    selected = indices.to(device=expansion.arc_ids.device, dtype=torch.int64)
    positions = expansion.frontier_positions[selected]
    counts = torch.bincount(positions, minlength=frontier_count)
    offsets64 = torch.cat(
        (
            torch.zeros(
                1,
                dtype=torch.int64,
                device=expansion.arc_ids.device,
            ),
            torch.cumsum(counts, dim=0),
        )
    )
    return FrontierExpansion(
        arc_ids=expansion.arc_ids[selected],
        edge_ids=expansion.edge_ids[selected],
        source_node_ids=expansion.source_node_ids[selected],
        destination_node_ids=expansion.destination_node_ids[selected],
        frontier_positions=positions,
        arc_offsets=offsets64.to(torch.int32),
    )


def candidate_control_features(
    hypotheses: HypothesisBatch,
    expansion: FrontierExpansion,
    state: ControllerState,
    *,
    config: SparseControllerConfig,
    search_limit: int,
    context_limit: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the sole candidate-control representation for every mode."""

    count = expansion.total_arcs
    if count == 0:
        return torch.empty(
            (0, 6),
            dtype=dtype,
            device=hypotheses.device,
        )
    parents = expansion.frontier_positions.to(torch.int64)
    depth = hypotheses.depths[parents].to(dtype)
    remaining_search = max(0, search_limit - state.arcs_scored)
    remaining_context = max(0, context_limit - state.contexts_read)
    constant = torch.tensor(
        [
            state.round_index / max(1, config.max_rounds),
            remaining_search / max(1, search_limit),
            remaining_context / max(1, context_limit),
            float(remaining_search == 0),
            float(remaining_context == 0),
        ],
        dtype=dtype,
        device=hypotheses.device,
    )
    return torch.cat(
        (
            depth[:, None] / max(1, config.max_depth),
            constant.expand(count, -1),
        ),
        dim=1,
    )


def termination_control_features(
    batch: PackedProgramBatch,
    hypotheses: HypothesisBatch,
    state: ControllerState,
    *,
    config: SparseControllerConfig,
    search_limit: int,
    context_limit: int,
) -> torch.Tensor:
    """Build post-transition controls with fixed runtime normalisation."""

    features = torch.zeros(
        (batch.graph_count, 6),
        dtype=hypotheses.path_state.dtype,
        device=batch.device,
    )
    features[:, 0] = state.round_index / max(1, config.max_rounds)
    features[:, 1] = min(state.arcs_scored, search_limit) / max(1, search_limit)
    features[:, 2] = min(state.contexts_read, context_limit) / max(
        1, context_limit
    )
    features[:, 3] = float(hypotheses.count == 0)
    features[:, 4] = float(state.search_budget_exhausted)
    features[:, 5] = float(state.context_budget_exhausted)
    return features


def _validate_action_indices(
    name: str,
    indices: torch.Tensor,
    *,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    resolved = indices.to(device=device, dtype=torch.int64)
    if resolved.ndim != 1:
        raise ValueError(f"{name} candidate indices must be one-dimensional")
    if resolved.numel() and (
        bool((resolved < 0).any().item())
        or bool((resolved >= count).any().item())
    ):
        raise IndexError(f"{name} candidate index is out of range")
    if resolved.numel() != torch.unique(resolved).numel():
        raise ValueError(f"{name} candidate indices must be unique")
    return resolved


def _source(fraction: float, randomizer: random.Random) -> ActionSource:
    if fraction <= 0.0:
        return ActionSource.MODEL
    if fraction >= 1.0:
        return ActionSource.ORACLE
    return (
        ActionSource.ORACLE
        if randomizer.random() < fraction
        else ActionSource.MODEL
    )


def _filtered_stable_selection(
    proposal: ControllerProposal,
    eligible_indices: torch.Tensor,
    priorities: torch.Tensor,
    *,
    width: int,
    per_node: int,
) -> torch.Tensor:
    if width <= 0 or eligible_indices.numel() == 0:
        return torch.empty(
            0,
            dtype=torch.int64,
            device=proposal.expansion.arc_ids.device,
        )
    sliced = _slice_expansion(
        proposal.expansion,
        eligible_indices,
        frontier_count=proposal.expansion.arc_offsets.numel() - 1,
    )
    selected_local = stable_candidate_selection(
        sliced,
        priorities[eligible_indices],
        frontier_width=width,
        hypotheses_per_node=per_node,
    )
    return eligible_indices[selected_local]


def _filtered_stable_evidence_selection(
    proposal: ControllerProposal,
    eligible_indices: torch.Tensor,
    priorities: torch.Tensor,
    *,
    width: int,
    state: ControllerState,
) -> torch.Tensor:
    """Select unique logical evidence actions without a destination cap."""

    device = proposal.expansion.arc_ids.device
    eligible = eligible_indices.to(device=device, dtype=torch.int64)
    if width <= 0 or eligible.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    if state.evidence_ledger:
        recorded_edges = torch.tensor(
            sorted({entry.edge_id for entry in state.evidence_ledger}),
            dtype=proposal.expansion.edge_ids.dtype,
            device=device,
        )
        repeated = (
            proposal.expansion.edge_ids[eligible, None]
            == recorded_edges[None, :]
        ).any(dim=1)
        eligible = eligible[~repeated]
        if eligible.numel() == 0:
            return eligible

    sliced = _slice_expansion(
        proposal.expansion,
        eligible,
        frontier_count=proposal.expansion.arc_offsets.numel() - 1,
    )
    local_order = _stable_priority_order(
        sliced,
        priorities[eligible],
    )
    ordered = eligible[local_order]
    ordered_edges = proposal.expansion.edge_ids[ordered].to(torch.int64)

    grouped_permutation = torch.argsort(ordered_edges, stable=True)
    grouped = ordered[grouped_permutation]
    grouped_edges = ordered_edges[grouped_permutation]
    first = torch.ones(
        grouped.numel(),
        dtype=torch.bool,
        device=device,
    )
    first[1:] = grouped_edges[1:] != grouped_edges[:-1]
    unique_candidates = grouped[first]

    priority_rank = torch.full(
        (proposal.expansion.total_arcs,),
        ordered.numel(),
        dtype=torch.int64,
        device=device,
    )
    priority_rank[ordered] = torch.arange(
        ordered.numel(),
        dtype=torch.int64,
        device=device,
    )
    unique_candidates = unique_candidates[
        torch.argsort(priority_rank[unique_candidates], stable=True)
    ]
    return unique_candidates[:width]


class SparseWavefrontController:
    """One packed controller transition shared by training and inference."""

    def __init__(self, config: SparseControllerConfig | None = None) -> None:
        self.config = config or SparseControllerConfig()

    def resolved_limits(
        self,
        batch: PackedProgramBatch,
    ) -> tuple[int, int, int]:
        if batch.graph_count == 1:
            case = batch.cases[0]
            search = min(self.config.search_budget, case.search_budget)
            context = min(
                self.config.context_read_budget,
                case.context_budget,
            )
        else:
            search = self.config.search_budget
            context = self.config.context_read_budget
        return search, context, self.config.evidence_selection_budget

    # Backwards-compatible private helper used by older callers.
    def _limits(self, batch: PackedProgramBatch) -> tuple[int, int]:
        search, context, _ = self.resolved_limits(batch)
        return search, context

    def _candidate_control(
        self,
        hypotheses: HypothesisBatch,
        expansion: FrontierExpansion,
        state: ControllerState,
        *,
        dtype: torch.dtype,
        search_limit: int,
        context_limit: int,
    ) -> torch.Tensor:
        return candidate_control_features(
            hypotheses,
            expansion,
            state,
            config=self.config,
            search_limit=search_limit,
            context_limit=context_limit,
            dtype=dtype,
        )

    def _termination_control(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        state: ControllerState,
        *,
        search_limit: int,
        context_limit: int,
    ) -> torch.Tensor:
        return termination_control_features(
            batch,
            hypotheses,
            state,
            config=self.config,
            search_limit=search_limit,
            context_limit=context_limit,
        )

    def propose(
        self,
        model: CandidateScorerBase,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        state: ControllerState,
    ) -> ControllerProposal:
        """Expand CSR and score candidates with the canonical controls."""

        full_expansion = batch.graph.topology.expand_frontier(
            hypotheses.node_ids,
            validate_ids=False,
        )
        search_limit, context_limit, evidence_limit = self.resolved_limits(batch)
        remaining_search = max(0, search_limit - state.arcs_scored)
        evaluated_count = min(full_expansion.total_arcs, remaining_search)
        search_truncated = evaluated_count != full_expansion.total_arcs
        if search_truncated:
            expansion = _slice_expansion(
                full_expansion,
                torch.arange(
                    evaluated_count,
                    dtype=torch.int64,
                    device=batch.device,
                ),
                frontier_count=hypotheses.count,
            )
        else:
            expansion = full_expansion
        controls = candidate_control_features(
            hypotheses,
            expansion,
            state,
            config=self.config,
            search_limit=search_limit,
            context_limit=context_limit,
            dtype=hypotheses.path_state.dtype,
        )
        outputs = model.score_candidates(
            batch,
            hypotheses,
            expansion,
            evidence,
            controls,
            round_index=state.round_index,
        )
        candidate_graph_ids = batch.graph.topology.node_graph_ids[
            expansion.source_node_ids.to(torch.int64)
        ]
        null_expansion_logits = None
        if self.config.expansion_policy == "learned_null":
            if not model.config.use_null_expansion:
                raise ValueError(
                    "learned_null expansion requires a model null-action head"
                )
            current_control = termination_control_features(
                batch,
                hypotheses,
                state,
                config=self.config,
                search_limit=search_limit,
                context_limit=context_limit,
            )
            null_expansion_logits = model.null_expansion_logits(
                batch,
                hypotheses,
                evidence,
                current_control,
            )
        if expansion.total_arcs:
            parents = expansion.frontier_positions.to(torch.int64)
            depth_eligible = (
                hypotheses.depths[parents].to(torch.int64) + 1
                <= self.config.max_depth
            )
        else:
            depth_eligible = torch.empty(
                0,
                dtype=torch.bool,
                device=batch.device,
            )
        return ControllerProposal(
            full_expansion=full_expansion,
            expansion=expansion,
            candidate_control_features=controls,
            candidate_outputs=outputs,
            depth_eligible=depth_eligible,
            search_limit=search_limit,
            context_limit=context_limit,
            evidence_limit=evidence_limit,
            full_arc_count=full_expansion.total_arcs,
            search_truncated=search_truncated,
            candidate_graph_ids=candidate_graph_ids,
            null_expansion_logits=null_expansion_logits,
        )

    def choose_actions(
        self,
        proposal: ControllerProposal,
        *,
        supervision: _StateSupervision | None,
        state: ControllerState,
        schedule: ActionSchedule,
        randomizer: random.Random,
        resolved_sources: dict[str, ActionSource] | None = None,
        context_candidate_indices: torch.Tensor | None = None,
    ) -> ControllerActions:
        """Choose four independently scheduled action sources."""

        candidates = None if supervision is None else supervision.candidates
        sources = (
            {
                "frontier": _source(schedule.frontier, randomizer),
                "context": _source(schedule.context, randomizer),
                "evidence": _source(schedule.evidence, randomizer),
                "termination": _source(schedule.termination, randomizer),
            }
            if resolved_sources is None
            else dict(resolved_sources)
        )
        if set(sources) != {
            "frontier",
            "context",
            "evidence",
            "termination",
        }:
            raise ValueError("resolved action sources are incomplete")
        if candidates is None and any(
            source is ActionSource.ORACLE for source in sources.values()
        ):
            raise ValueError("oracle action source requires state supervision")

        outputs = proposal.candidate_outputs
        device = proposal.expansion.arc_ids.device
        remaining_context = max(
            0,
            proposal.context_limit - state.contexts_read,
        )
        remaining_evidence = max(
            0,
            proposal.evidence_limit - state.evidence_selected,
        )

        if context_candidate_indices is not None:
            context = _validate_action_indices(
                "context",
                context_candidate_indices,
                count=proposal.expansion.total_arcs,
                device=device,
            )
            if context.numel() > remaining_context:
                raise ValueError("context actions exceed the remaining budget")
        elif sources["context"] is ActionSource.ORACLE:
            assert candidates is not None
            context = torch.nonzero(
                candidates.context_has_value,
                as_tuple=False,
            ).flatten()[:remaining_context]
        else:
            eligible = torch.nonzero(
                torch.sigmoid(outputs.context_logits)
                >= self.config.context_threshold,
                as_tuple=False,
            ).flatten()
            order = _stable_priority_order(
                proposal.expansion,
                outputs.context_logits,
            )
            eligible_mask = torch.zeros(
                proposal.expansion.total_arcs,
                dtype=torch.bool,
                device=device,
            )
            if eligible.numel():
                eligible_mask[eligible] = True
            context = order[eligible_mask[order]][:remaining_context]

        if sources["evidence"] is ActionSource.ORACLE:
            assert candidates is not None
            evidence_eligible = torch.nonzero(
                candidates.include_as_evidence,
                as_tuple=False,
            ).flatten()
            evidence = _filtered_stable_evidence_selection(
                proposal,
                evidence_eligible,
                -candidates.remaining_cost,
                width=remaining_evidence,
                state=state,
            )
        else:
            evidence_eligible = torch.nonzero(
                torch.sigmoid(outputs.evidence_logits)
                >= self.config.evidence_threshold,
                as_tuple=False,
            ).flatten()
            evidence = _filtered_stable_evidence_selection(
                proposal,
                evidence_eligible,
                outputs.evidence_logits,
                width=remaining_evidence,
                state=state,
            )

        if sources["frontier"] is ActionSource.ORACLE:
            assert candidates is not None
            frontier_eligible = torch.nonzero(
                candidates.acceptable & proposal.depth_eligible,
                as_tuple=False,
            ).flatten()
            frontier = _filtered_stable_selection(
                proposal,
                frontier_eligible,
                -candidates.remaining_cost,
                width=self.config.frontier_width,
                per_node=self.config.hypotheses_per_node,
            )
        else:
            model_eligible = (
                proposal.depth_eligible
                & (
                    torch.sigmoid(outputs.expand_logits)
                    >= self.config.expand_threshold
                )
            )
            if self.config.expansion_policy == "learned_null":
                if proposal.null_expansion_logits is None:
                    raise ValueError(
                        "learned_null proposal is missing null-action logits"
                    )
                choose_null = (
                    proposal.null_expansion_logits[
                        proposal.candidate_graph_ids.to(torch.int64)
                    ]
                    >= 0
                )
                model_eligible = model_eligible & ~choose_null
            frontier_eligible = torch.nonzero(
                model_eligible,
                as_tuple=False,
            ).flatten()
            combined = outputs.priority_logits + F.logsigmoid(
                outputs.expand_logits
            )
            frontier = _filtered_stable_selection(
                proposal,
                frontier_eligible,
                combined,
                width=self.config.frontier_width,
                per_node=self.config.hypotheses_per_node,
            )

        return ControllerActions(
            frontier_candidate_indices=frontier,
            context_candidate_indices=context,
            evidence_candidate_indices=evidence,
            frontier_source=sources["frontier"],
            context_source=sources["context"],
            evidence_source=sources["evidence"],
            termination_source=sources["termination"],
        )

    def select_actions(
        self,
        model: CandidateScorerBase,
        batch: PackedProgramBatch,
        proposal: ControllerProposal,
        *,
        supervision: _StateSupervision | None,
        state: ControllerState,
        schedule: ActionSchedule,
        randomizer: random.Random,
    ) -> ControllerActionSelection:
        """Choose context first, then use refined outputs for later actions."""

        preliminary = self.choose_actions(
            proposal,
            supervision=supervision,
            state=state,
            schedule=schedule,
            randomizer=randomizer,
        )
        refined_outputs = model.refine_with_context(
            batch,
            proposal.expansion,
            proposal.candidate_outputs,
            preliminary.context_candidate_indices,
        )
        refined_proposal = replace(
            proposal,
            candidate_outputs=refined_outputs,
            pre_context_outputs=proposal.candidate_outputs,
            context_refined=True,
        )
        resolved_sources = {
            "frontier": preliminary.frontier_source,
            "context": preliminary.context_source,
            "evidence": preliminary.evidence_source,
            "termination": preliminary.termination_source,
        }
        actions = self.choose_actions(
            refined_proposal,
            supervision=supervision,
            state=state,
            schedule=schedule,
            randomizer=randomizer,
            resolved_sources=resolved_sources,
            context_candidate_indices=(
                preliminary.context_candidate_indices
            ),
        )
        return ControllerActionSelection(
            proposal=refined_proposal,
            actions=actions,
        )

    def apply(
        self,
        model: CandidateScorerBase,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        state: ControllerState,
        proposal: ControllerProposal,
        actions: ControllerActions,
    ) -> ControllerTransition:
        """Apply context, evidence, frontier, ledgers, and budgets once."""

        count = proposal.expansion.total_arcs
        context_indices = _validate_action_indices(
            "context",
            actions.context_candidate_indices,
            count=count,
            device=batch.device,
        )
        evidence_indices = _validate_action_indices(
            "evidence",
            actions.evidence_candidate_indices,
            count=count,
            device=batch.device,
        )
        frontier_indices = _validate_action_indices(
            "frontier",
            actions.frontier_candidate_indices,
            count=count,
            device=batch.device,
        )
        remaining_context = max(
            0,
            proposal.context_limit - state.contexts_read,
        )
        remaining_evidence = max(
            0,
            proposal.evidence_limit - state.evidence_selected,
        )
        if context_indices.numel() > remaining_context:
            raise ValueError("context actions exceed the remaining budget")
        if evidence_indices.numel() > remaining_evidence:
            raise ValueError("evidence actions exceed the remaining budget")
        if frontier_indices.numel() > self.config.frontier_width:
            raise ValueError("frontier actions exceed configured width")
        if frontier_indices.numel() and not bool(
            proposal.depth_eligible[frontier_indices].all().item()
        ):
            raise ValueError("frontier actions include a depth-ineligible candidate")

        refined = (
            proposal.candidate_outputs
            if proposal.context_refined
            else model.refine_with_context(
                batch,
                proposal.expansion,
                proposal.candidate_outputs,
                context_indices,
            )
        )
        context_flags = torch.zeros(
            count,
            dtype=torch.bool,
            device=batch.device,
        )
        if context_indices.numel():
            context_flags[context_indices] = True

        context_entries = list(state.context_ledger)
        for candidate_index in context_indices.tolist():
            frontier_position = int(
                proposal.expansion.frontier_positions[candidate_index].item()
            )
            context_entries.append(
                ContextLedgerEntry(
                    node_id=int(
                        proposal.expansion.destination_node_ids[
                            candidate_index
                        ].item()
                    ),
                    edge_id=int(
                        proposal.expansion.edge_ids[candidate_index].item()
                    ),
                    arc_id=int(
                        proposal.expansion.arc_ids[candidate_index].item()
                    ),
                    round_index=state.round_index,
                    frontier_position=frontier_position,
                    parent_trace_id=int(
                        hypotheses.parent_trace_ids[frontier_position].item()
                    ),
                )
            )

        evidence_entries = list(state.evidence_ledger)
        if evidence_indices.numel():
            evidence_parents = proposal.expansion.frontier_positions[
                evidence_indices
            ].to(torch.int64)
            evidence_graph_ids = hypotheses.graph_ids[evidence_parents]
            evidence = model.update_evidence(
                evidence,
                refined.next_path_state[evidence_indices].mean(dim=1),
                evidence_graph_ids,
            )
            for candidate_index in evidence_indices.tolist():
                frontier_position = int(
                    proposal.expansion.frontier_positions[candidate_index].item()
                )
                evidence_entries.append(
                    EvidenceLedgerEntry(
                        node_id=int(
                            proposal.expansion.destination_node_ids[
                                candidate_index
                            ].item()
                        ),
                        edge_id=int(
                            proposal.expansion.edge_ids[candidate_index].item()
                        ),
                        arc_id=int(
                            proposal.expansion.arc_ids[candidate_index].item()
                        ),
                        round_index=state.round_index,
                        frontier_position=frontier_position,
                        parent_trace_id=int(
                            hypotheses.parent_trace_ids[
                                frontier_position
                            ].item()
                        ),
                        context_read=bool(
                            context_flags[candidate_index].item()
                        ),
                    )
                )

        trace_entries = list(state.trace_ledger)
        parent_trace_ids: list[int] = []
        for candidate_index in frontier_indices.tolist():
            frontier_position = int(
                proposal.expansion.frontier_positions[candidate_index].item()
            )
            trace_entries.append(
                TraceLedgerEntry(
                    node_id=int(
                        proposal.expansion.destination_node_ids[
                            candidate_index
                        ].item()
                    ),
                    edge_id=int(
                        proposal.expansion.edge_ids[candidate_index].item()
                    ),
                    arc_id=int(
                        proposal.expansion.arc_ids[candidate_index].item()
                    ),
                    round_index=state.round_index,
                    frontier_position=frontier_position,
                    parent_trace_id=int(
                        hypotheses.parent_trace_ids[frontier_position].item()
                    ),
                    context_read=bool(
                        context_flags[candidate_index].item()
                    ),
                )
            )
            parent_trace_ids.append(len(trace_entries) - 1)

        if frontier_indices.numel():
            parents = proposal.expansion.frontier_positions[
                frontier_indices
            ].to(torch.int64)
            next_hypotheses = HypothesisBatch(
                node_ids=proposal.expansion.destination_node_ids[
                    frontier_indices
                ],
                graph_ids=hypotheses.graph_ids[parents],
                path_state=refined.next_path_state[frontier_indices],
                scores=(
                    hypotheses.scores[parents]
                    + refined.priority_logits[frontier_indices]
                ),
                depths=hypotheses.depths[parents] + 1,
                parent_trace_ids=torch.tensor(
                    parent_trace_ids,
                    dtype=torch.int64,
                    device=batch.device,
                ),
                incoming_arc_ids=proposal.expansion.arc_ids[
                    frontier_indices
                ],
                incoming_edge_ids=proposal.expansion.edge_ids[
                    frontier_indices
                ],
                context_read=context_flags[frontier_indices],
            ).validate()
        else:
            next_hypotheses = model.empty_hypotheses(batch.device)

        contexts_read = state.contexts_read + int(context_indices.numel())
        evidence_selected = state.evidence_selected + int(
            evidence_indices.numel()
        )
        arcs_scored = state.arcs_scored + count
        search_exhausted = (
            arcs_scored >= proposal.search_limit
            or proposal.search_truncated
            or proposal.search_limit == 0
        )
        context_exhausted = (
            contexts_read >= proposal.context_limit
            or proposal.context_limit == 0
        )
        evidence_exhausted = (
            evidence_selected >= proposal.evidence_limit
            or proposal.evidence_limit == 0
        )
        had_arcs = proposal.full_arc_count > 0
        no_depth_eligible = (
            proposal.expansion.total_arcs > 0
            and not bool(proposal.depth_eligible.any().item())
        )
        deliberate_empty = (
            had_arcs
            and frontier_indices.numel() == 0
            and not search_exhausted
            and not no_depth_eligible
        )
        next_state = ControllerState(
            round_index=state.round_index + 1,
            arcs_scored=arcs_scored,
            contexts_read=contexts_read,
            evidence_selected=evidence_selected,
            search_budget_exhausted=search_exhausted,
            context_budget_exhausted=context_exhausted,
            evidence_budget_exhausted=evidence_exhausted,
            frontier_empty=next_hypotheses.count == 0,
            last_expansion_had_arcs=had_arcs,
            last_expansion_truncated=proposal.search_truncated,
            deliberate_empty_frontier=deliberate_empty,
            depth_exhausted=no_depth_eligible,
            trace_ledger=tuple(trace_entries),
            context_ledger=tuple(context_entries),
            evidence_ledger=tuple(evidence_entries),
        )
        termination_control = termination_control_features(
            batch,
            next_hypotheses,
            next_state,
            config=self.config,
            search_limit=proposal.search_limit,
            context_limit=proposal.context_limit,
        )
        return ControllerTransition(
            proposal=replace(proposal, candidate_outputs=refined),
            actions=replace(
                actions,
                frontier_candidate_indices=frontier_indices,
                context_candidate_indices=context_indices,
                evidence_candidate_indices=evidence_indices,
            ),
            refined_outputs=refined,
            next_hypotheses=next_hypotheses,
            next_evidence=evidence,
            next_controller_state=next_state,
            termination_control=termination_control,
        )

    def transition(
        self,
        model: CandidateScorerBase,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        state: ControllerState,
        *,
        supervision: _StateSupervision | None,
        schedule: ActionSchedule,
        randomizer: random.Random,
    ) -> ControllerTransition:
        proposal = self.propose(model, batch, hypotheses, evidence, state)
        selection = self.select_actions(
            model,
            batch,
            proposal,
            supervision=supervision,
            state=state,
            schedule=schedule,
            randomizer=randomizer,
        )
        return self.apply(
            model,
            batch,
            hypotheses,
            evidence,
            state,
            selection.proposal,
            selection.actions,
        )

    def step(
        self,
        model: CandidateScorerBase,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        state: ControllerState,
    ) -> ControllerTransition:
        """Compatibility runtime step using model actions only."""

        return self.transition(
            model,
            batch,
            hypotheses,
            evidence,
            state,
            supervision=None,
            schedule=ActionSchedule.model_only(),
            randomizer=random.Random(0),
        )

    @staticmethod
    def _empty_fallback(state: ControllerState) -> TerminationDecision:
        if (
            state.search_budget_exhausted
            or state.depth_exhausted
            or state.deliberate_empty_frontier
        ):
            return TerminationDecision.UNKNOWN_INCOMPLETE
        return TerminationDecision.UNKNOWN_ABSENT

    def execute_termination(
        self,
        output: torch.Tensor | TerminationOutput,
        transition: ControllerTransition,
    ) -> tuple[TerminationDecision, ...]:
        """Execute model termination with the canonical empty-state fallback."""

        logits = output.logits if isinstance(output, TerminationOutput) else output
        if (
            isinstance(output, TerminationOutput)
            and output.evidence_sufficient_logits is not None
        ):
            if (
                output.useful_work_remaining_logits is None
                or output.answer_supported_logits is None
                or output.unknown_logits is None
            ):
                raise ValueError("factorized termination output is incomplete")
            state = transition.next_controller_state
            exact_stop = (
                state.frontier_empty
                or state.search_budget_exhausted
                or state.depth_exhausted
                or state.round_index >= self.config.max_rounds
            )
            sufficient = output.evidence_sufficient_logits >= 0
            useful = output.useful_work_remaining_logits >= 0
            answer = output.answer_supported_logits >= 0
            unknown_indices = output.unknown_logits.argmax(dim=-1)
            decisions = []
            for graph_id in range(logits.shape[0]):
                stop_allowed = (
                    bool(sufficient[graph_id].item())
                    or not bool(useful[graph_id].item())
                    or exact_stop
                )
                if not stop_allowed:
                    decisions.append(TerminationDecision.CONTINUE)
                elif (
                    bool(sufficient[graph_id].item())
                    and bool(answer[graph_id].item())
                ):
                    decisions.append(TerminationDecision.ANSWER)
                else:
                    decisions.append(
                        _TERMINATION_CLASSES[
                            int(unknown_indices[graph_id].item()) + 2
                        ]
                    )
            return tuple(decisions)
        decisions = tuple(
            _TERMINATION_CLASSES[index]
            for index in logits.argmax(dim=-1).tolist()
        )
        if transition.next_hypotheses.count == 0:
            decisions = tuple(
                self._empty_fallback(transition.next_controller_state)
                if decision is TerminationDecision.CONTINUE
                else decision
                for decision in decisions
            )
        return decisions

    def run(
        self,
        model: CandidateScorerBase,
        batch: PackedProgramBatch,
        *,
        execution_policy: ControllerExecutionPolicy | None = None,
    ) -> ControllerResult:
        policy = execution_policy or ControllerExecutionPolicy.learned()
        round_limit = policy.resolve_round_limit(
            batch,
            configured_max_rounds=self.config.max_rounds,
        )
        hypotheses = model.initial_hypotheses(batch)
        evidence = model.initial_evidence(batch)
        state = ControllerState.initial()
        arc_trace: list[tuple[int, ...]] = []
        diagnostics: list[ActionDiagnostic] = []
        round_records: list[ControllerRoundRecord] = []
        termination = tuple(
            TerminationDecision.CONTINUE for _ in range(batch.graph_count)
        )
        final_logits = evidence.new_zeros((batch.graph_count, len(_TERMINATION_CLASSES)))
        randomizer = random.Random(0)
        for round_offset in range(round_limit):
            hypotheses = apply_path_state_intervention(
                model,
                batch,
                hypotheses,
                intervention=policy.path_state_intervention,
                round_index=state.round_index,
                seed=policy.intervention_seed,
            )
            round_hypotheses = hypotheses
            round_state = state
            proposal = self.propose(model, batch, hypotheses, evidence, state)
            selection = self.select_actions(
                model,
                batch,
                proposal,
                supervision=None,
                state=state,
                schedule=ActionSchedule.model_only(),
                randomizer=randomizer,
            )
            proposal = selection.proposal
            actions = selection.actions
            transition = self.apply(
                model,
                batch,
                hypotheses,
                evidence,
                state,
                proposal,
                actions,
            )
            arc_trace.append(
                tuple(
                    transition.expansion.arc_ids[
                        transition.selected_candidate_indices
                    ].tolist()
                )
            )
            hypotheses = transition.next_hypotheses
            evidence = transition.next_evidence
            state = transition.next_controller_state
            evaluate_termination = (
                policy.horizon_mode is HorizonMode.LEARNED
                or round_offset == round_limit - 1
            )
            final_output: TerminationOutput | None = None
            if evaluate_termination:
                final_output = model.termination_output(
                    batch,
                    hypotheses,
                    evidence,
                    transition.termination_control,
                )
                final_logits = final_output.logits
                termination = self.execute_termination(
                    final_output,
                    transition,
                )
            else:
                termination = tuple(
                    TerminationDecision.CONTINUE
                    for _ in range(batch.graph_count)
                )
            round_records.append(
                ControllerRoundRecord(
                    hypotheses=round_hypotheses,
                    controller_state=round_state,
                    proposal=proposal,
                    actions=actions,
                    transition=transition,
                    termination_output=final_output,
                    termination=termination,
                )
            )
            diagnostics.append(
                ActionDiagnostic(
                    round_index=state.round_index - 1,
                    frontier_source=actions.frontier_source,
                    context_source=actions.context_source,
                    evidence_source=actions.evidence_source,
                    termination_source=actions.termination_source,
                    frontier_candidate_indices=tuple(
                        actions.frontier_candidate_indices.tolist()
                    ),
                    context_candidate_indices=tuple(
                        actions.context_candidate_indices.tolist()
                    ),
                    evidence_candidate_indices=tuple(
                        actions.evidence_candidate_indices.tolist()
                    ),
                    executed_termination=termination[0],
                )
            )
            if (
                policy.horizon_mode is HorizonMode.LEARNED
                and all(
                    decision is not TerminationDecision.CONTINUE
                    for decision in termination
                )
            ):
                break
        if all(
            decision is TerminationDecision.CONTINUE
            for decision in termination
        ):
            termination = tuple(
                TerminationDecision.UNKNOWN_INCOMPLETE
                if decision is TerminationDecision.CONTINUE
                else decision
                for decision in termination
            )
        return ControllerResult(
            hypotheses=hypotheses,
            evidence=evidence,
            termination=termination,
            selected_arc_trace=tuple(arc_trace),
            trace_ledger=state.trace_ledger,
            context_ledger=state.context_ledger,
            evidence_ledger=state.evidence_ledger,
            action_diagnostics=tuple(diagnostics),
            final_termination_logits=final_logits,
            rounds=state.round_index,
            arcs_scored=state.arcs_scored,
            contexts_read=state.contexts_read,
            round_records=tuple(round_records),
        )
