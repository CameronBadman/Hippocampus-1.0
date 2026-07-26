from __future__ import annotations

from dataclasses import dataclass

from .schema import GraphProgramCase, ProgramFamily, TerminationDecision


@dataclass(frozen=True, slots=True)
class VerificationReport:
    valid: bool
    errors: tuple[str, ...]

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"graph-program verification failed:\n{joined}")


def _directed_edge_matches(
    case: GraphProgramCase,
    edge_id: int,
    source: int,
    destination: int,
) -> bool:
    if not 0 <= edge_id < len(case.edges):
        return False
    edge = case.edges[edge_id]
    return (
        edge.source_node == source and edge.destination_node == destination
    ) or (
        edge.bidirectional
        and edge.destination_node == source
        and edge.source_node == destination
    )


def _valid_arc_exists(
    case: GraphProgramCase,
    source: int,
    destination: int,
) -> bool:
    return any(
        edge.valid
        and (
            edge.source_node == source
            and edge.destination_node == destination
            or edge.bidirectional
            and edge.destination_node == source
            and edge.source_node == destination
        )
        for edge in case.edges
    )


def verify_case(case: GraphProgramCase) -> VerificationReport:
    """Mechanically verify topology, trace, evidence, and termination contracts."""

    errors: list[str] = []
    node_count = len(case.nodes)
    node_ids = (
        *case.start_nodes,
        *case.answer_nodes,
        *case.evidence_nodes,
    )
    if any(node_id < 0 or node_id >= node_count for node_id in node_ids):
        errors.append("start, answer, or evidence node is out of range")
    for edge_id in case.evidence_edge_ids:
        if not 0 <= edge_id < len(case.edges):
            errors.append("exact evidence edge is out of range")
        elif case.edges[edge_id].destination_node not in case.evidence_nodes:
            errors.append(
                "exact evidence edge must terminate at an evidence node"
            )

    latent_nodes = [node.latent_id for node in case.nodes]
    latent_edges = [edge.latent_id for edge in case.edges]
    if len(set(latent_nodes)) != len(latent_nodes):
        errors.append("latent node IDs must be unique")
    if len(set(latent_edges)) != len(latent_edges):
        errors.append("latent edge IDs must be unique")

    for edge_id, edge in enumerate(case.edges):
        if not 0 <= edge.source_node < node_count:
            errors.append(f"edge {edge_id} source is out of range")
        if not 0 <= edge.destination_node < node_count:
            errors.append(f"edge {edge_id} destination is out of range")

    if not case.trace.rounds:
        errors.append("oracle trace must contain at least one round")
    for round_index, round_ in enumerate(case.trace.rounds):
        if any(node < 0 or node >= node_count for node in round_.frontier_nodes):
            errors.append(f"round {round_index} frontier contains an invalid node")
        frontier = set(round_.frontier_nodes)
        for candidate_index, candidate in enumerate(round_.candidates):
            prefix = f"round {round_index} candidate {candidate_index}"
            if candidate.source_node not in frontier:
                errors.append(f"{prefix} source is not active in the frontier")
            if not _directed_edge_matches(
                case,
                candidate.edge_id,
                candidate.source_node,
                candidate.destination_node,
            ):
                errors.append(f"{prefix} does not resolve to its declared arc")
                continue
            edge = case.edges[candidate.edge_id]
            if candidate.acceptable and not edge.valid:
                errors.append(f"{prefix} marks an invalid edge acceptable")
            if candidate.context_has_value:
                node = case.nodes[candidate.destination_node]
                if not node.context_atoms:
                    errors.append(f"{prefix} requests an empty context")
                elif not set(node.context_atoms) - set(node.summary_atoms):
                    errors.append(
                        f"{prefix} context has no observation absent from summary"
                    )
            if candidate.include_as_evidence and (
                candidate.destination_node not in case.evidence_nodes
            ):
                errors.append(
                    f"{prefix} includes a node outside the exact evidence set"
                )
            if (
                candidate.include_as_evidence
                and case.evidence_edge_ids
                and candidate.edge_id not in case.evidence_edge_ids
            ):
                errors.append(
                    f"{prefix} includes an edge outside the exact evidence set"
                )

    if case.evidence_edge_ids:
        labelled_evidence_edges = {
            candidate.edge_id
            for round_ in case.trace.rounds
            for candidate in round_.candidates
            if candidate.include_as_evidence
        }
        if labelled_evidence_edges != set(case.evidence_edge_ids):
            errors.append(
                "exact evidence edge set disagrees with candidate labels"
            )

    for path_index, path in enumerate(case.trace.valid_paths):
        if len(path) < 2:
            errors.append(f"valid path {path_index} must contain at least one arc")
            continue
        if path[0] not in case.start_nodes:
            errors.append(f"valid path {path_index} does not begin at a start node")
        for source, destination in zip(path[:-1], path[1:], strict=True):
            if not _valid_arc_exists(case, source, destination):
                errors.append(
                    f"valid path {path_index} uses invalid arc {source}->{destination}"
                )

    is_answer = case.termination.decision is TerminationDecision.ANSWER
    if is_answer != bool(case.answer_nodes):
        errors.append("answerability and answer node set disagree")
    if case.termination.answer_nodes != case.answer_nodes:
        errors.append("termination answer set disagrees with case answer set")
    if case.trace.rounds and case.trace.rounds[-1].termination != case.termination:
        errors.append("final oracle round termination disagrees with the case")
    if is_answer and not case.trace.valid_paths:
        errors.append("answerable cases require at least one valid path")

    if case.family is ProgramFamily.REACHABILITY and is_answer:
        path_lengths = {len(path) for path in case.trace.valid_paths}
        if len(path_lengths) != 1:
            errors.append("reachability oracle paths must all be shortest/equal length")
        if any(path[-1] not in case.answer_nodes for path in case.trace.valid_paths):
            errors.append("reachability paths must terminate at an answer node")

    if case.family is ProgramFamily.LATEST_VALID:
        context_targets = [
            candidate
            for round_ in case.trace.rounds
            for candidate in round_.candidates
            if candidate.context_has_value
        ]
        if not context_targets:
            errors.append("latest-valid cases require positive context-read targets")

    if case.family is ProgramFamily.CORROBORATION and is_answer:
        if len(case.evidence_nodes) < 2:
            errors.append("corroboration answers require at least two evidence nodes")

    if (
        case.termination.decision is TerminationDecision.UNKNOWN_CONFLICT
        and len(case.evidence_nodes) < 2
    ):
        errors.append("conflict termination requires conflicting evidence")

    return VerificationReport(valid=not errors, errors=tuple(errors))
