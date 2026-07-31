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
class EvidenceRequirement:
    """Supervisor-only exact identity of one required evidence action."""

    edge_id: int | None
    source_node: int | None
    destination_node: int

    @property
    def edge_specific(self) -> bool:
        return self.edge_id is not None


@dataclass(frozen=True, slots=True)
class StateSupervision:
    """Exact labels for the actual packed rollout state."""

    candidates: CandidateSupervision
    recoverable: bool
    remaining_evidence_nodes: tuple[int, ...]
    scored_evidence_nodes: tuple[int, ...]
    remaining_evidence_requirements: tuple[EvidenceRequirement, ...] = ()
    scored_evidence_requirements: tuple[EvidenceRequirement, ...] = ()


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
        self.required_evidence = self._build_evidence_requirements()
        self._context_nodes = {
            target.destination_node
            for target in self._targets.values()
            if target.context_has_value
        }

    def _build_evidence_requirements(
        self,
    ) -> tuple[EvidenceRequirement, ...]:
        if self.case.evidence_edge_ids:
            requirements = []
            for edge_id in self.case.evidence_edge_ids:
                edge = self.case.edges[edge_id]
                requirements.append(
                    EvidenceRequirement(
                        edge_id=edge_id,
                        source_node=edge.source_node,
                        destination_node=edge.destination_node,
                    )
                )
            return tuple(requirements)
        return tuple(
            EvidenceRequirement(
                edge_id=None,
                source_node=None,
                destination_node=node_id,
            )
            for node_id in self.case.evidence_nodes
        )

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
            requirement.destination_node
            for requirement in self._accumulated_requirements(state)
        }

    def _entry_matches_requirement(
        self,
        *,
        node_id: int,
        edge_id: int,
        requirement: EvidenceRequirement,
    ) -> bool:
        destination = self._local_node(node_id)
        if destination != requirement.destination_node:
            return False
        if requirement.edge_id is None:
            return True
        return edge_id - self.edge_offset == requirement.edge_id

    def _accumulated_requirements(
        self,
        state: ControllerState,
    ) -> set[EvidenceRequirement]:
        return {
            requirement
            for requirement in self.required_evidence
            if any(
                self._entry_matches_requirement(
                    node_id=entry.node_id,
                    edge_id=entry.edge_id,
                    requirement=requirement,
                )
                for entry in state.evidence_ledger
            )
        }

    def requirement_for_candidate(
        self,
        *,
        edge_id: int,
        source_node: int,
        destination_node: int,
    ) -> EvidenceRequirement | None:
        local_edge = edge_id - self.edge_offset
        local_source = self._local_node(source_node)
        local_destination = self._local_node(destination_node)
        for requirement in self.required_evidence:
            if requirement.destination_node != local_destination:
                continue
            if requirement.edge_id is None:
                return requirement
            if (
                requirement.edge_id == local_edge
                and requirement.source_node == local_source
            ):
                return requirement
        return None

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

    def _requirement_is_reachable(
        self,
        requirement: EvidenceRequirement,
        hypotheses: HypothesisBatch,
        state: ControllerState,
    ) -> bool:
        if requirement in self._accumulated_requirements(state):
            return True
        if hypotheses.count == 0:
            return False

        remaining_rounds = self.config.max_rounds - state.round_index
        search_limit = min(self.config.search_budget, self.case.search_budget)
        remaining_search = search_limit - state.arcs_scored
        if remaining_rounds <= 0 or remaining_search <= 0:
            return False

        context_limit = min(
            self.config.context_read_budget,
            self.case.context_budget,
        )
        remaining_context = context_limit - state.contexts_read
        contexts_read = self._read_contexts(state)
        frontier_nodes = tuple(int(node) for node in hypotheses.node_ids.tolist())
        frontier_depths = tuple(int(depth) for depth in hypotheses.depths.tolist())
        queue: deque[
            tuple[tuple[int, ...], tuple[int, ...], int, int]
        ] = deque(
            (
                (
                    frontier_nodes,
                    frontier_depths,
                    0,
                    remaining_search,
                ),
            )
        )
        visited: set[tuple[int, int, int, int]] = set()
        topology = self.batch.graph.topology

        while queue:
            nodes, depths, rounds_used, budget = queue.popleft()
            if rounds_used >= remaining_rounds or budget <= 0 or not nodes:
                continue
            node_tensor = torch.tensor(
                nodes,
                dtype=torch.int32,
                device=self.batch.device,
            )
            expansion = topology.expand_frontier(
                node_tensor,
                validate_ids=False,
            )
            scored_count = min(expansion.total_arcs, budget)
            if scored_count == 0:
                continue
            for candidate_index in range(scored_count):
                edge_id = int(expansion.edge_ids[candidate_index].item())
                source = int(
                    expansion.source_node_ids[candidate_index].item()
                )
                destination = int(
                    expansion.destination_node_ids[candidate_index].item()
                )
                candidate_requirement = self.requirement_for_candidate(
                    edge_id=edge_id,
                    source_node=source,
                    destination_node=destination,
                )
                if candidate_requirement == requirement:
                    target = self._target_for_occurrence(
                        edge_id,
                        source,
                        destination,
                    )
                    needs_context = (
                        target.context_has_value
                        and requirement.destination_node not in contexts_read
                    )
                    if not needs_context or remaining_context > 0:
                        return True

                target = self._target_for_occurrence(
                    edge_id,
                    source,
                    destination,
                )
                if not target.acceptable:
                    continue
                parent_position = int(
                    expansion.frontier_positions[candidate_index].item()
                )
                next_depth = depths[parent_position] + 1
                if next_depth > self.config.max_depth:
                    continue
                next_budget = budget - scored_count
                next_round = rounds_used + 1
                key = (destination, next_depth, next_round, next_budget)
                if key in visited:
                    continue
                visited.add(key)
                queue.append(
                    ((destination,), (next_depth,), next_round, next_budget)
                )
        return False

    def reachable_evidence_requirements(
        self,
        hypotheses: HypothesisBatch,
        state: ControllerState,
    ) -> tuple[EvidenceRequirement, ...]:
        """Return exact action identities reachable within controller limits."""

        return tuple(
            requirement
            for requirement in self.required_evidence
            if self._requirement_is_reachable(
                requirement,
                hypotheses,
                state,
            )
        )

    def _completion_is_reachable(
        self,
        hypotheses: HypothesisBatch,
        state: ControllerState,
    ) -> bool:
        missing_requirements = (
            set(self.required_evidence)
            - self._accumulated_requirements(state)
        )
        if not missing_requirements:
            return self.case.answerable or (
                self.case.termination.decision
                is TerminationDecision.UNKNOWN_CONFLICT
            )
        reachable = set(
            self.reachable_evidence_requirements(hypotheses, state)
        )
        return missing_requirements.issubset(reachable)

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
        accumulated_requirements = self._accumulated_requirements(state)
        contexts_read = self._read_contexts(state)
        missing_requirements = (
            set(self.required_evidence) - accumulated_requirements
        )
        missing = {
            requirement.destination_node
            for requirement in missing_requirements
        }
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
        plausible_negative: list[bool] = []
        scored_evidence: set[int] = set()
        scored_requirements: set[EvidenceRequirement] = set()
        for target, edge_id, source, destination_global in zip(
            targets,
            proposal.expansion.edge_ids.tolist(),
            proposal.expansion.source_node_ids.tolist(),
            proposal.expansion.destination_node_ids.tolist(),
            strict=True,
        ):
            destination = target.destination_node
            requirement = self.requirement_for_candidate(
                edge_id=edge_id,
                source_node=source,
                destination_node=destination_global,
            )
            preserves_completion = (
                target.acceptable
                and (
                    requirement in missing_requirements
                    or any(
                        self._distance(destination, required) is not None
                        for required in missing
                    )
                )
            )
            include = (
                target.include_as_evidence
                and requirement in missing_requirements
            )
            if include:
                scored_evidence.add(destination)
                assert requirement is not None
                scored_requirements.add(requirement)
            acceptable.append(preserves_completion)
            context.append(
                target.context_has_value
                and destination not in contexts_read
            )
            evidence.append(include)
            remaining.append(target.remaining_cost)
            support.append(target.support)
            conflict.append(target.conflict)
            plausible_negative.append(
                not include
                and (
                    preserves_completion
                    or destination in missing
                    or target.support > 0
                    or target.conflict > 0
                )
            )

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
            evidence_plausible_negative=torch.tensor(
                plausible_negative,
                dtype=torch.bool,
                device=device,
            ),
        )
        return StateSupervision(
            candidates=candidates,
            recoverable=self._completion_is_reachable(hypotheses, state),
            remaining_evidence_nodes=tuple(sorted(missing)),
            scored_evidence_nodes=tuple(sorted(scored_evidence)),
            remaining_evidence_requirements=tuple(
                sorted(
                    missing_requirements,
                    key=lambda item: (
                        -1 if item.edge_id is None else item.edge_id,
                        -1 if item.source_node is None else item.source_node,
                        item.destination_node,
                    ),
                )
            ),
            scored_evidence_requirements=tuple(
                sorted(
                    scored_requirements,
                    key=lambda item: (
                        -1 if item.edge_id is None else item.edge_id,
                        -1 if item.source_node is None else item.source_node,
                        item.destination_node,
                    ),
                )
            ),
        )

    def termination_target(
        self,
        transition: ControllerTransition,
    ) -> TerminationTarget:
        state = transition.next_controller_state
        accumulated = self._accumulated_requirements(state)
        required = set(self.required_evidence)
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
