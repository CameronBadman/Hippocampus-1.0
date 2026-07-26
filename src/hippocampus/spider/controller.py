from __future__ import annotations

from dataclasses import dataclass

import torch

from ..programs.batching import PackedProgramBatch
from ..programs.schema import TerminationDecision
from ..topology import FrontierExpansion
from .config import SparseControllerConfig
from .hypothesis import HypothesisBatch
from .model import CandidateScorerBase
from .types import CandidateOutputs


_TERMINATION_CLASSES = (
    TerminationDecision.CONTINUE,
    TerminationDecision.ANSWER,
    TerminationDecision.UNKNOWN_ABSENT,
    TerminationDecision.UNKNOWN_CONFLICT,
    TerminationDecision.UNKNOWN_INCOMPLETE,
    TerminationDecision.UNKNOWN_UNSUPPORTED,
)


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
    search_budget_exhausted: bool
    context_budget_exhausted: bool
    trace_ledger: tuple[TraceLedgerEntry, ...]
    evidence_ledger: tuple[EvidenceLedgerEntry, ...]

    @classmethod
    def initial(cls) -> "ControllerState":
        return cls(
            round_index=0,
            arcs_scored=0,
            contexts_read=0,
            search_budget_exhausted=False,
            context_budget_exhausted=False,
            trace_ledger=(),
            evidence_ledger=(),
        )


@dataclass(frozen=True, slots=True)
class ControllerStep:
    expansion: FrontierExpansion
    outputs: CandidateOutputs
    selected_candidate_indices: torch.Tensor
    context_candidate_indices: torch.Tensor
    next_hypotheses: HypothesisBatch
    evidence: torch.Tensor
    state: ControllerState


@dataclass(frozen=True, slots=True)
class ControllerResult:
    hypotheses: HypothesisBatch
    evidence: torch.Tensor
    termination: tuple[TerminationDecision, ...]
    selected_arc_trace: tuple[tuple[int, ...], ...]
    trace_ledger: tuple[TraceLedgerEntry, ...]
    evidence_ledger: tuple[EvidenceLedgerEntry, ...]
    rounds: int
    arcs_scored: int
    contexts_read: int


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


class SparseWavefrontController:
    def __init__(self, config: SparseControllerConfig | None = None) -> None:
        self.config = config or SparseControllerConfig()

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
        count = expansion.total_arcs
        if count == 0:
            return torch.empty(
                (0, 6),
                dtype=dtype,
                device=hypotheses.device,
            )
        parents = expansion.frontier_positions.to(torch.int64)
        depth = hypotheses.depths[parents].to(dtype)
        search_remaining = max(
            0,
            search_limit - state.arcs_scored,
        )
        context_remaining = max(
            0,
            context_limit - state.contexts_read,
        )
        constant = torch.tensor(
            [
                state.round_index / max(1, self.config.max_rounds),
                search_remaining / max(1, search_limit),
                context_remaining / max(1, context_limit),
                float(state.search_budget_exhausted),
                float(state.context_budget_exhausted),
            ],
            dtype=dtype,
            device=hypotheses.device,
        )
        return torch.cat(
            (
                depth[:, None] / max(1, self.config.max_depth),
                constant.expand(count, -1),
            ),
            dim=1,
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
        features = torch.zeros(
            (batch.graph_count, 6),
            dtype=hypotheses.path_state.dtype,
            device=batch.device,
        )
        features[:, 0] = state.round_index / max(1, self.config.max_rounds)
        features[:, 1] = state.arcs_scored / max(1, search_limit)
        features[:, 2] = state.contexts_read / max(
            1,
            context_limit,
        )
        features[:, 3] = float(hypotheses.count == 0)
        features[:, 4] = float(state.search_budget_exhausted)
        features[:, 5] = float(state.context_budget_exhausted)
        return features

    def _limits(self, batch: PackedProgramBatch) -> tuple[int, int]:
        if batch.graph_count == 1:
            case = batch.cases[0]
            return (
                min(self.config.search_budget, case.search_budget),
                min(self.config.context_read_budget, case.context_budget),
            )
        return (
            self.config.search_budget,
            self.config.context_read_budget,
        )

    def step(
        self,
        model: CandidateScorerBase,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        evidence: torch.Tensor,
        state: ControllerState,
    ) -> ControllerStep:
        full_expansion = batch.graph.topology.expand_frontier(
            hypotheses.node_ids,
            validate_ids=False,
        )
        search_limit, context_limit = self._limits(batch)
        remaining_search = max(
            0,
            search_limit - state.arcs_scored,
        )
        evaluated_count = min(full_expansion.total_arcs, remaining_search)
        if evaluated_count != full_expansion.total_arcs:
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
        budget_exhausted = (
            state.arcs_scored + expansion.total_arcs
            >= search_limit
        )
        control = self._candidate_control(
            hypotheses,
            expansion,
            state,
            dtype=hypotheses.path_state.dtype,
            search_limit=search_limit,
            context_limit=context_limit,
        )
        outputs = model.score_candidates(
            batch,
            hypotheses,
            expansion,
            evidence,
            control,
            round_index=state.round_index,
        )

        remaining_context = max(
            0,
            context_limit - state.contexts_read,
        )
        if expansion.total_arcs and remaining_context:
            context_order = _stable_priority_order(
                expansion,
                outputs.context_logits,
            )
            positive = (
                torch.sigmoid(outputs.context_logits[context_order]) >= 0.5
            )
            context_indices = context_order[positive][:remaining_context]
            outputs = model.refine_with_context(
                batch,
                expansion,
                outputs,
                context_indices,
            )
        else:
            context_indices = torch.empty(
                0,
                dtype=torch.int64,
                device=batch.device,
            )

        if expansion.total_arcs:
            parents = expansion.frontier_positions.to(torch.int64)
            eligible = (
                hypotheses.depths[parents].to(torch.int64) + 1
                <= self.config.max_depth
            )
            eligible_indices = torch.nonzero(
                eligible,
                as_tuple=False,
            ).flatten()
            if eligible_indices.numel():
                eligible_expansion = _slice_expansion(
                    expansion,
                    eligible_indices,
                    frontier_count=hypotheses.count,
                )
                combined_priority = (
                    outputs.priority_logits[eligible_indices]
                    + torch.nn.functional.logsigmoid(
                        outputs.expand_logits[eligible_indices]
                    )
                )
                selected_local = stable_candidate_selection(
                    eligible_expansion,
                    combined_priority,
                    frontier_width=self.config.frontier_width,
                    hypotheses_per_node=self.config.hypotheses_per_node,
                )
                selected = eligible_indices[selected_local]
            else:
                selected = eligible_indices
        else:
            selected = torch.empty(
                0,
                dtype=torch.int64,
                device=batch.device,
            )

        context_flags = torch.zeros(
            expansion.total_arcs,
            dtype=torch.bool,
            device=batch.device,
        )
        if context_indices.numel():
            context_flags = context_flags.index_fill(0, context_indices, True)
        trace_entries = list(state.trace_ledger)
        evidence_entries = list(state.evidence_ledger)
        parent_trace_ids: list[int] = []
        selected_list = selected.tolist()
        for candidate_index in selected_list:
            frontier_position = int(
                expansion.frontier_positions[candidate_index].item()
            )
            parent_trace_id = int(
                hypotheses.parent_trace_ids[frontier_position].item()
            )
            entry = TraceLedgerEntry(
                node_id=int(
                    expansion.destination_node_ids[candidate_index].item()
                ),
                edge_id=int(expansion.edge_ids[candidate_index].item()),
                arc_id=int(expansion.arc_ids[candidate_index].item()),
                round_index=state.round_index,
                frontier_position=frontier_position,
                parent_trace_id=parent_trace_id,
                context_read=bool(context_flags[candidate_index].item()),
            )
            trace_entries.append(entry)
            parent_trace_ids.append(len(trace_entries) - 1)

        if selected.numel():
            parent_positions = expansion.frontier_positions[selected].to(torch.int64)
            graph_ids = hypotheses.graph_ids[parent_positions]
            next_hypotheses = HypothesisBatch(
                node_ids=expansion.destination_node_ids[selected],
                graph_ids=graph_ids,
                path_state=outputs.next_path_state[selected],
                scores=(
                    hypotheses.scores[parent_positions]
                    + outputs.priority_logits[selected]
                ),
                depths=hypotheses.depths[parent_positions] + 1,
                parent_trace_ids=torch.tensor(
                    parent_trace_ids,
                    dtype=torch.int64,
                    device=batch.device,
                ),
                incoming_arc_ids=expansion.arc_ids[selected],
                incoming_edge_ids=expansion.edge_ids[selected],
                context_read=context_flags[selected],
            ).validate()
            evidence_mask = (
                torch.sigmoid(outputs.evidence_logits[selected])
                >= self.config.evidence_threshold
            )
            evidence_candidates = selected[evidence_mask]
            if evidence_candidates.numel():
                evidence_parent_positions = expansion.frontier_positions[
                    evidence_candidates
                ].to(torch.int64)
                evidence_graph_ids = hypotheses.graph_ids[
                    evidence_parent_positions
                ]
                evidence = model.update_evidence(
                    evidence,
                    outputs.next_path_state[evidence_candidates].mean(dim=1),
                    evidence_graph_ids,
                )
                for candidate_index in evidence_candidates.tolist():
                    frontier_position = int(
                        expansion.frontier_positions[candidate_index].item()
                    )
                    evidence_entries.append(
                        EvidenceLedgerEntry(
                            node_id=int(
                                expansion.destination_node_ids[
                                    candidate_index
                                ].item()
                            ),
                            edge_id=int(
                                expansion.edge_ids[candidate_index].item()
                            ),
                            arc_id=int(
                                expansion.arc_ids[candidate_index].item()
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
        else:
            next_hypotheses = model.empty_hypotheses(batch.device)

        contexts_read = state.contexts_read + int(context_indices.numel())
        next_state = ControllerState(
            round_index=state.round_index + 1,
            arcs_scored=state.arcs_scored + expansion.total_arcs,
            contexts_read=contexts_read,
            search_budget_exhausted=budget_exhausted,
            context_budget_exhausted=contexts_read >= context_limit,
            trace_ledger=tuple(trace_entries),
            evidence_ledger=tuple(evidence_entries),
        )
        return ControllerStep(
            expansion=expansion,
            outputs=outputs,
            selected_candidate_indices=selected,
            context_candidate_indices=context_indices,
            next_hypotheses=next_hypotheses,
            evidence=evidence,
            state=next_state,
        )

    def run(
        self,
        model: CandidateScorerBase,
        batch: PackedProgramBatch,
    ) -> ControllerResult:
        hypotheses = model.initial_hypotheses(batch)
        evidence = model.initial_evidence(batch)
        state = ControllerState.initial()
        search_limit, context_limit = self._limits(batch)
        arc_trace: list[tuple[int, ...]] = []
        termination = tuple(
            TerminationDecision.CONTINUE
            for _ in range(batch.graph_count)
        )
        for _ in range(self.config.max_rounds):
            step = self.step(
                model,
                batch,
                hypotheses,
                evidence,
                state,
            )
            arc_trace.append(
                tuple(
                    step.expansion.arc_ids[
                        step.selected_candidate_indices
                    ].tolist()
                )
            )
            hypotheses = step.next_hypotheses
            evidence = step.evidence
            state = step.state
            if hypotheses.count == 0:
                decision = (
                    TerminationDecision.UNKNOWN_INCOMPLETE
                    if state.search_budget_exhausted
                    else TerminationDecision.UNKNOWN_ABSENT
                )
                termination = tuple(decision for _ in range(batch.graph_count))
                break
            logits = model.termination_logits(
                batch,
                hypotheses,
                evidence,
                self._termination_control(
                    batch,
                    hypotheses,
                    state,
                    search_limit=search_limit,
                    context_limit=context_limit,
                ),
            )
            termination = tuple(
                _TERMINATION_CLASSES[index]
                for index in logits.argmax(dim=-1).tolist()
            )
            if all(
                decision is not TerminationDecision.CONTINUE
                for decision in termination
            ):
                break
        else:
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
            evidence_ledger=state.evidence_ledger,
            rounds=state.round_index,
            arcs_scored=state.arcs_scored,
            contexts_read=state.contexts_read,
        )
