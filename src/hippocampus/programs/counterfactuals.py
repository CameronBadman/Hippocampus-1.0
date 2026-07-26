from __future__ import annotations

import hashlib
from dataclasses import replace

from .schema import (
    CandidateTarget,
    CounterfactualKind,
    GraphProgramCase,
    Intervention,
    OracleRound,
    ParallelOracleTrace,
    ProgramFamily,
    ProgramNode,
    TerminationDecision,
    TerminationTarget,
)


def _decisive_edge_id(case: GraphProgramCase) -> int:
    for round_ in case.trace.rounds:
        for candidate in round_.candidates:
            if candidate.acceptable:
                return candidate.edge_id
    raise ValueError("case has no decisive acceptable edge")


def _unknown_after_edge_edit(
    case: GraphProgramCase,
    *,
    edges: tuple,
    changed_latent_id: int,
    kind: CounterfactualKind,
) -> GraphProgramCase:
    start_frontier = case.trace.rounds[0].frontier_nodes
    candidates = tuple(
        CandidateTarget(
            edge_id=edge_id,
            source_node=edge.source_node,
            destination_node=edge.destination_node,
            acceptable=False,
            priority_tier=1,
            remaining_cost=1.0,
        )
        for edge_id, edge in enumerate(edges)
        if edge.source_node in start_frontier
    )
    termination = TerminationTarget(TerminationDecision.UNKNOWN_ABSENT)
    digest = hashlib.sha256(
        f"{case.case_id}|{kind.value}|{changed_latent_id}".encode()
    ).hexdigest()
    return replace(
        case,
        case_id=f"{case.family.value}-counterfactual-{digest[:20]}",
        view_id=f"counterfactual-{kind.value}",
        edges=edges,
        answer_nodes=(),
        evidence_nodes=(),
        trace=ParallelOracleTrace(
            rounds=(OracleRound(start_frontier, candidates, termination),),
            valid_paths=(),
        ),
        termination=termination,
        intervention=Intervention(kind, (changed_latent_id,)),
    )


def make_counterfactual(
    case: GraphProgramCase,
    *,
    kind: CounterfactualKind | str,
) -> GraphProgramCase:
    """Apply one exact intervention and recompute its supervisor targets.

    The initial edge-removal/reversal path is intentionally strict: it is only
    accepted for an answerable lookup or single-path reachability case where
    the edited edge is decisive.
    """

    resolved = CounterfactualKind(kind)
    if resolved in {
        CounterfactualKind.ALTER_TEMPORAL_VALUE,
        CounterfactualKind.ADD_NEWER_CONFLICT,
    }:
        return _latest_counterfactual(case, resolved)
    if resolved is CounterfactualKind.INVALIDATE_SOURCE:
        return _invalidate_corroborating_source(case)
    if resolved not in {
        CounterfactualKind.REMOVE_DECISIVE_EDGE,
        CounterfactualKind.DISCONNECT_ONLY_PATH,
        CounterfactualKind.REVERSE_DECISIVE_EDGE,
        CounterfactualKind.REPLACE_ENDPOINT,
    }:
        raise ValueError(f"unsupported counterfactual kind {resolved.value}")
    if not case.answerable:
        raise ValueError("counterfactual source case must be answerable")
    if case.family not in {ProgramFamily.LOOKUP, ProgramFamily.REACHABILITY}:
        raise ValueError("edge counterfactual requires lookup or reachability")
    if (
        case.family is ProgramFamily.REACHABILITY
        and len(case.trace.valid_paths) != 1
    ):
        raise ValueError("edge counterfactual requires one decisive path")

    decisive_id = _decisive_edge_id(case)
    decisive = case.edges[decisive_id]
    if resolved in {
        CounterfactualKind.REMOVE_DECISIVE_EDGE,
        CounterfactualKind.DISCONNECT_ONLY_PATH,
    }:
        edges = tuple(
            edge for edge_id, edge in enumerate(case.edges) if edge_id != decisive_id
        )
    elif resolved is CounterfactualKind.REVERSE_DECISIVE_EDGE:
        edges = tuple(
            replace(
                edge,
                source_node=edge.destination_node,
                destination_node=edge.source_node,
            )
            if edge_id == decisive_id
            else edge
            for edge_id, edge in enumerate(case.edges)
        )
    else:
        replacement = next(
            node_id
            for node_id in range(len(case.nodes))
            if node_id
            not in {
                decisive.source_node,
                decisive.destination_node,
                *case.answer_nodes,
            }
        )
        edges = tuple(
            replace(edge, destination_node=replacement)
            if edge_id == decisive_id
            else edge
            for edge_id, edge in enumerate(case.edges)
        )
    return _unknown_after_edge_edit(
        case,
        edges=edges,
        changed_latent_id=decisive.latent_id,
        kind=resolved,
    )


def _replace_context_scalar(
    node: ProgramNode,
    *,
    atom_index: int,
    value: float,
) -> ProgramNode:
    context = list(node.context_atoms)
    context[atom_index] = replace(context[atom_index], scalar=value)
    return replace(node, context_atoms=tuple(context))


def _retarget_trace(
    case: GraphProgramCase,
    *,
    decision: TerminationDecision,
    answer_nodes: tuple[int, ...],
    evidence_nodes: tuple[int, ...],
    acceptable_nodes: set[int],
    conflict_nodes: set[int] | None = None,
    context_nodes: set[int] | None = None,
) -> ParallelOracleTrace:
    conflict_targets = conflict_nodes or set()
    context_targets = context_nodes or set()
    rounds: list[OracleRound] = []
    for round_index, round_ in enumerate(case.trace.rounds):
        candidates = tuple(
            replace(
                candidate,
                acceptable=candidate.destination_node in acceptable_nodes,
                priority_tier=(
                    0
                    if candidate.destination_node in acceptable_nodes
                    else 1
                ),
                context_has_value=(
                    candidate.context_has_value
                    or candidate.destination_node in context_targets
                ),
                include_as_evidence=(
                    candidate.destination_node in evidence_nodes
                ),
                support=(
                    1.0
                    if candidate.destination_node in answer_nodes
                    else 0.0
                ),
                conflict=(
                    1.0
                    if candidate.destination_node in conflict_targets
                    else 0.0
                ),
            )
            for candidate in round_.candidates
        )
        termination = (
            TerminationTarget(decision, answer_nodes)
            if round_index == len(case.trace.rounds) - 1
            else TerminationTarget(TerminationDecision.CONTINUE)
        )
        rounds.append(OracleRound(round_.frontier_nodes, candidates, termination))
    destinations = evidence_nodes or answer_nodes
    valid_paths = tuple(
        (case.start_nodes[0], destination)
        for destination in destinations
    )
    return ParallelOracleTrace(tuple(rounds), valid_paths)


def _latest_counterfactual(
    case: GraphProgramCase,
    kind: CounterfactualKind,
) -> GraphProgramCase:
    if case.family is not ProgramFamily.LATEST_VALID or not case.answerable:
        raise ValueError("temporal counterfactual requires answerable latest-valid case")
    record_nodes = tuple(
        dict.fromkeys(
            candidate.destination_node
            for round_ in case.trace.rounds
            for candidate in round_.candidates
            if candidate.context_has_value
        )
    )
    if len(record_nodes) < 2:
        raise ValueError("temporal counterfactual requires multiple records")

    def time_atom(node_id: int) -> tuple[int, float]:
        scalars = [
            (index, float(atom.scalar))
            for index, atom in enumerate(case.nodes[node_id].context_atoms)
            if atom.scalar is not None
        ]
        if not scalars:
            raise ValueError("record context has no temporal scalar")
        return max(scalars, key=lambda item: item[1])

    current = case.answer_nodes[0]
    current_atom, current_time = time_atom(current)
    alternatives = sorted(
        (
            (time_atom(node_id)[1], node_id)
            for node_id in record_nodes
            if node_id != current
        ),
        reverse=True,
    )
    alternative = alternatives[0][1]
    nodes = list(case.nodes)
    if kind is CounterfactualKind.ALTER_TEMPORAL_VALUE:
        nodes[current] = _replace_context_scalar(
            nodes[current],
            atom_index=current_atom,
            value=min(0.0, alternatives[-1][0] - 1.0),
        )
        answer_nodes = (alternative,)
        evidence_nodes = answer_nodes
        decision = TerminationDecision.ANSWER
        conflict_nodes: set[int] = set()
        changed_node = current
    else:
        alternative_atom, _ = time_atom(alternative)
        nodes[alternative] = _replace_context_scalar(
            nodes[alternative],
            atom_index=alternative_atom,
            value=current_time,
        )
        answer_nodes = ()
        evidence_nodes = (current, alternative)
        decision = TerminationDecision.UNKNOWN_CONFLICT
        conflict_nodes = set(evidence_nodes)
        changed_node = alternative
    trace = _retarget_trace(
        case,
        decision=decision,
        answer_nodes=answer_nodes,
        evidence_nodes=evidence_nodes,
        acceptable_nodes=set(evidence_nodes),
        conflict_nodes=conflict_nodes,
    )
    digest = hashlib.sha256(
        f"{case.case_id}|{kind.value}|{case.nodes[changed_node].latent_id}".encode()
    ).hexdigest()
    return replace(
        case,
        case_id=f"latest-counterfactual-{digest[:20]}",
        view_id=f"counterfactual-{kind.value}",
        nodes=tuple(nodes),
        answer_nodes=answer_nodes,
        evidence_nodes=evidence_nodes,
        trace=trace,
        termination=TerminationTarget(decision, answer_nodes),
        intervention=Intervention(
            kind,
            (case.nodes[changed_node].latent_id,),
        ),
    )


def _invalidate_corroborating_source(
    case: GraphProgramCase,
) -> GraphProgramCase:
    if case.family is not ProgramFamily.CORROBORATION or not case.answerable:
        raise ValueError(
            "source invalidation requires an answerable corroboration case"
        )
    invalidated = case.answer_nodes[0]
    node = case.nodes[invalidated]
    scalar_atoms = [
        index
        for index, atom in enumerate(node.context_atoms)
        if atom.scalar is not None and atom.scalar > 0
    ]
    if not scalar_atoms:
        raise ValueError("corroborating source has no observable validity scalar")
    nodes = list(case.nodes)
    nodes[invalidated] = _replace_context_scalar(
        node,
        atom_index=scalar_atoms[0],
        value=-1.0,
    )
    remaining = tuple(
        node_id for node_id in case.answer_nodes if node_id != invalidated
    )
    decision = TerminationDecision.UNKNOWN_ABSENT
    trace = _retarget_trace(
        case,
        decision=decision,
        answer_nodes=(),
        evidence_nodes=remaining,
        acceptable_nodes=set(remaining),
        context_nodes={invalidated},
    )
    digest = hashlib.sha256(
        f"{case.case_id}|invalidate|{node.latent_id}".encode()
    ).hexdigest()
    return replace(
        case,
        case_id=f"corroboration-counterfactual-{digest[:20]}",
        view_id="counterfactual-invalidate_source",
        nodes=tuple(nodes),
        answer_nodes=(),
        evidence_nodes=remaining,
        trace=trace,
        termination=TerminationTarget(decision),
        intervention=Intervention(
            CounterfactualKind.INVALIDATE_SOURCE,
            (node.latent_id,),
        ),
    )
