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
    if resolved not in {
        CounterfactualKind.REMOVE_DECISIVE_EDGE,
        CounterfactualKind.DISCONNECT_ONLY_PATH,
        CounterfactualKind.REVERSE_DECISIVE_EDGE,
        CounterfactualKind.REPLACE_ENDPOINT,
    }:
        raise NotImplementedError(
            f"{resolved.value} requires family-specific scalar/source regeneration"
        )
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
