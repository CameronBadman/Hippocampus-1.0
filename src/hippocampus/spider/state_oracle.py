from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from ..programs.batching import PackedProgramBatch
from ..programs.schema import (
    CandidateTarget,
    GraphProgramCase,
    TerminationDecision,
    TerminationTarget,
)
from .config import SparseControllerConfig
from .controller import ControllerProposal, ControllerState, ControllerTransition
from .hypothesis import HypothesisBatch
from .losses import CandidateSupervision


@dataclass(frozen=True, slots=True)
class StateSupervision:
    """Exact labels for the actual packed rollout state."""

    candidates: CandidateSupervision
    recoverable: bool
    remaining_evidence_nodes: tuple[int, ...]
    scored_evidence_nodes: tuple[int, ...]


class StateOracle:
    """Supervisor-only oracle for arbitrary controller states.

    The oracle never requires equality with a recorded parallel frontier.
    Recorded acceptable transitions define the program-valid transition
    relation; current hypotheses, exact ledgers, and remaining budgets define
    whether a completion is still possible.
    """

    def __init__(
        self,
        case: GraphProgramCase,
        batch: PackedProgramBatch,
        config: SparseControllerConfig,
    ) -> None:
        if batch.graph_count != 1 or batch.cases[0].case_id != case.case_id:
            raise ValueError("StateOracle requires its aligned singleton batch")
        self.case = case
        self.batch = batch
        self.config = config
        self.node_offset = int(batch.graph.topology.graph_node_ptr[0].item())
        self.edge_offset = int(batch.graph.topology.graph_edge_ptr[0].item())
        self._targets = self._build_target_map()
        self._adjacency = self._build_program_adjacency()
        self._context_nodes = {
            target.destination_node
            for target in self._targets.values()
            if target.context_has_value
        }

    def _build_target_map(
        self,
    ) -> dict[tuple[int, int, int], CandidateTarget]:
        targets: dict[tuple[int, int, int], CandidateTarget] = {}
        for round_ in self.case.trace.rounds:
            for target in round_.candidates:
                key = (
                    target.edge_id,
                    target.source_node,
                    target.destination_node,
                )
                previous = targets.get(key)
                if previous is None:
                    targets[key] = target
                    continue
                # Repeated wavefront occurrences must agree semantically. Keep
                # the most permissive valid action while retaining the shortest
                # remaining cost.
                targets[key] = CandidateTarget(
                    edge_id=target.edge_id,
                    source_node=target.source_node,
                    destination_node=target.destination_node,
                    acceptable=previous.acceptable or target.acceptable,
                    priority_tier=min(
                        previous.priority_tier,
                        target.priority_tier,
                    ),
                    remaining_cost=min(
                        previous.remaining_cost,
                        target.remaining_cost,
                    ),
                    context_has_value=(
                        previous.context_has_value
                        or target.context_has_value
                    ),
                    include_as_evidence=(
                        previous.include_as_evidence
                        or target.include_as_evidence
                    ),
                    support=max(previous.support, target.support),
                    conflict=max(previous.conflict, target.conflict),
                )
        return targets

    def _build_program_adjacency(self) -> dict[int, tuple[int, ...]]:
        mutable: dict[int, set[int]] = {}
        for target in self._targets.values():
            if target.acceptable:
                mutable.setdefault(target.source_node, set()).add(
                    target.destination_node
                )
        return {
            source: tuple(sorted(destinations))
            for source, destinations in mutable.items()
        }

    def _local_node(self, global_node: int) -> int:
        return global_node - self.node_offset

    def _accumulated_evidence(
        self,
        state: ControllerState,
    ) -> set[int]:
        return {
            self._local_node(entry.node_id)
            for entry in state.evidence_ledger
        }

    def _read_contexts(self, state: ControllerState) -> set[int]:
        return {
            self._local_node(entry.node_id)
            for entry in state.context_ledger
        }

    def _distance(self, source: int, destination: int) -> int | None:
        if source == destination:
            return 0
        queue: deque[tuple[int, int]] = deque(((source, 0),))
        visited = {source}
        while queue:
            node, distance = queue.popleft()
            for neighbour in self._adjacency.get(node, ()):
                if neighbour == destination:
                    return distance + 1
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, distance + 1))
        return None

    def _completion_is_reachable(
        self,
        hypotheses: HypothesisBatch,
        state: ControllerState,
    ) -> bool:
        required = set(self.case.evidence_nodes)
        missing = required - self._accumulated_evidence(state)
        if not missing:
            return self.case.answerable or (
                self.case.termination.decision
                is TerminationDecision.UNKNOWN_CONFLICT
            )
        if hypotheses.count == 0:
            return False
        remaining_rounds = self.config.max_rounds - state.round_index
        search_limit = min(self.config.search_budget, self.case.search_budget)
        remaining_search = search_limit - state.arcs_scored
        context_limit = min(
            self.config.context_read_budget,
            self.case.context_budget,
        )
        remaining_context = context_limit - state.contexts_read
        if remaining_rounds <= 0 or remaining_search <= 0:
            return False
        contexts_read = self._read_contexts(state)
        needed_contexts = len(
            (missing & self._context_nodes) - contexts_read
        )
        if needed_contexts > remaining_context:
            return False

        local_nodes = [
            self._local_node(node) for node in hypotheses.node_ids.tolist()
        ]
        depths = hypotheses.depths.tolist()
        total_distance = 0
        for target in missing:
            feasible_distances = [
                distance
                for node, depth in zip(local_nodes, depths, strict=True)
                if (distance := self._distance(node, target)) is not None
                and distance <= remaining_rounds
                and depth + distance <= self.config.max_depth
            ]
            if not feasible_distances:
                return False
            total_distance += min(feasible_distances)
        return total_distance <= remaining_search

    def _target_for_occurrence(
        self,
        edge_id: int,
        source: int,
        destination: int,
    ) -> CandidateTarget:
        key = (
            edge_id - self.edge_offset,
            source - self.node_offset,
            destination - self.node_offset,
        )
        return self._targets.get(
            key,
            CandidateTarget(
                edge_id=key[0],
                source_node=key[1],
                destination_node=key[2],
                acceptable=False,
                priority_tier=1,
                remaining_cost=float(self.config.max_depth),
            ),
        )

    def label(
        self,
        proposal: ControllerProposal,
        hypotheses: HypothesisBatch,
        state: ControllerState,
    ) -> StateSupervision:
        accumulated = self._accumulated_evidence(state)
        contexts_read = self._read_contexts(state)
        missing = set(self.case.evidence_nodes) - accumulated
        targets = [
            self._target_for_occurrence(edge_id, source, destination)
            for edge_id, source, destination in zip(
                proposal.expansion.edge_ids.tolist(),
                proposal.expansion.source_node_ids.tolist(),
                proposal.expansion.destination_node_ids.tolist(),
                strict=True,
            )
        ]
        acceptable: list[bool] = []
        context: list[bool] = []
        evidence: list[bool] = []
        remaining: list[float] = []
        support: list[float] = []
        conflict: list[float] = []
        scored_evidence: set[int] = set()
        for target in targets:
            destination = target.destination_node
            preserves_completion = (
                target.acceptable
                and (
                    destination in missing
                    or any(
                        self._distance(destination, required) is not None
                        for required in missing
                    )
                )
            )
            include = (
                target.include_as_evidence
                and destination in missing
            )
            if include:
                scored_evidence.add(destination)
            acceptable.append(preserves_completion)
            context.append(
                target.context_has_value
                and destination not in contexts_read
            )
            evidence.append(include)
            remaining.append(target.remaining_cost)
            support.append(target.support)
            conflict.append(target.conflict)

        device = self.batch.device
        candidates = CandidateSupervision(
            acceptable=torch.tensor(
                acceptable,
                dtype=torch.bool,
                device=device,
            ),
            context_has_value=torch.tensor(
                context,
                dtype=torch.bool,
                device=device,
            ),
            include_as_evidence=torch.tensor(
                evidence,
                dtype=torch.bool,
                device=device,
            ),
            remaining_cost=torch.tensor(
                remaining,
                dtype=torch.float32,
                device=device,
            ),
            support=torch.tensor(
                support,
                dtype=torch.float32,
                device=device,
            ),
            conflict=torch.tensor(
                conflict,
                dtype=torch.float32,
                device=device,
            ),
        )
        return StateSupervision(
            candidates=candidates,
            recoverable=self._completion_is_reachable(hypotheses, state),
            remaining_evidence_nodes=tuple(sorted(missing)),
            scored_evidence_nodes=tuple(sorted(scored_evidence)),
        )

    def termination_target(
        self,
        transition: ControllerTransition,
    ) -> TerminationTarget:
        state = transition.next_controller_state
        accumulated = self._accumulated_evidence(state)
        required = set(self.case.evidence_nodes)
        expected = self.case.termination.decision

        if expected is TerminationDecision.UNKNOWN_UNSUPPORTED:
            return TerminationTarget(TerminationDecision.UNKNOWN_UNSUPPORTED)
        if self.case.answerable and required.issubset(accumulated):
            return TerminationTarget(
                TerminationDecision.ANSWER,
                self.case.answer_nodes,
            )
        if (
            expected is TerminationDecision.UNKNOWN_CONFLICT
            and required.issubset(accumulated)
        ):
            return TerminationTarget(TerminationDecision.UNKNOWN_CONFLICT)
        if self._completion_is_reachable(
            transition.next_hypotheses,
            state,
        ):
            return TerminationTarget(TerminationDecision.CONTINUE)
        if expected is TerminationDecision.UNKNOWN_ABSENT:
            return TerminationTarget(TerminationDecision.UNKNOWN_ABSENT)
        return TerminationTarget(TerminationDecision.UNKNOWN_INCOMPLETE)
