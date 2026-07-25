from __future__ import annotations

import hashlib
import random
from dataclasses import replace

from .schema import (
    CandidateTarget,
    GraphProgramCase,
    ObservableAtom,
    OracleRound,
    ParallelOracleTrace,
    ProgramEdge,
    ProgramNode,
    TerminationTarget,
)


def _map_atom(
    atom: ObservableAtom,
    symbol_map: dict[str, str],
) -> ObservableAtom:
    return ObservableAtom(
        symbols=tuple(symbol_map[symbol] for symbol in atom.symbols),
        scalar=atom.scalar,
    )


def make_equivalent_view(
    case: GraphProgramCase,
    *,
    seed: int,
) -> GraphProgramCase:
    """Re-key and reorder one case while preserving all latent truth."""

    rng = random.Random(seed)
    symbols = {
        symbol
        for atom in case.query_atoms
        for symbol in atom.symbols
    }
    for node in case.nodes:
        for atom in (*node.summary_atoms, *node.context_atoms):
            symbols.update(atom.symbols)
    for edge in case.edges:
        for atom in edge.atoms:
            symbols.update(atom.symbols)
    symbol_map = {
        symbol: f"view_{rng.getrandbits(64):016x}"
        for symbol in sorted(symbols)
    }

    old_node_order = list(range(len(case.nodes)))
    rng.shuffle(old_node_order)
    old_to_new_node = {
        old_node: new_node
        for new_node, old_node in enumerate(old_node_order)
    }
    new_nodes: list[ProgramNode] = []
    for old_node in old_node_order:
        source = case.nodes[old_node]
        summary = [_map_atom(atom, symbol_map) for atom in source.summary_atoms]
        context = [_map_atom(atom, symbol_map) for atom in source.context_atoms]
        rng.shuffle(summary)
        rng.shuffle(context)
        new_nodes.append(
            ProgramNode(
                latent_id=source.latent_id,
                summary_atoms=tuple(summary),
                context_atoms=tuple(context),
            )
        )

    old_edge_order = list(range(len(case.edges)))
    rng.shuffle(old_edge_order)
    old_to_new_edge = {
        old_edge: new_edge
        for new_edge, old_edge in enumerate(old_edge_order)
    }
    new_edges: list[ProgramEdge] = []
    for old_edge in old_edge_order:
        source = case.edges[old_edge]
        atoms = [_map_atom(atom, symbol_map) for atom in source.atoms]
        rng.shuffle(atoms)
        new_edges.append(
            ProgramEdge(
                latent_id=source.latent_id,
                source_node=old_to_new_node[source.source_node],
                destination_node=old_to_new_node[source.destination_node],
                atoms=tuple(atoms),
                bidirectional=source.bidirectional,
                valid=source.valid,
            )
        )

    def mapped_termination(target: TerminationTarget) -> TerminationTarget:
        return TerminationTarget(
            target.decision,
            tuple(old_to_new_node[node] for node in target.answer_nodes),
        )

    rounds: list[OracleRound] = []
    for round_ in case.trace.rounds:
        candidates = tuple(
            replace(
                candidate,
                edge_id=old_to_new_edge[candidate.edge_id],
                source_node=old_to_new_node[candidate.source_node],
                destination_node=old_to_new_node[candidate.destination_node],
            )
            for candidate in round_.candidates
        )
        rounds.append(
            OracleRound(
                frontier_nodes=tuple(
                    old_to_new_node[node] for node in round_.frontier_nodes
                ),
                candidates=candidates,
                termination=mapped_termination(round_.termination),
            )
        )

    mapped_query = [_map_atom(atom, symbol_map) for atom in case.query_atoms]
    rng.shuffle(mapped_query)
    digest = hashlib.sha256(
        f"{case.base_case_id}|equivalent|{seed}".encode()
    ).hexdigest()
    return GraphProgramCase(
        case_id=f"{case.family.value}-view-{digest[:20]}",
        base_case_id=case.base_case_id,
        view_id=f"equivalent-{seed}",
        seed=case.seed,
        family=case.family,
        nodes=tuple(new_nodes),
        edges=tuple(new_edges),
        query_atoms=tuple(mapped_query),
        start_nodes=tuple(old_to_new_node[node] for node in case.start_nodes),
        answer_nodes=tuple(old_to_new_node[node] for node in case.answer_nodes),
        evidence_nodes=tuple(old_to_new_node[node] for node in case.evidence_nodes),
        trace=ParallelOracleTrace(
            rounds=tuple(rounds),
            valid_paths=tuple(
                tuple(old_to_new_node[node] for node in path)
                for path in case.trace.valid_paths
            ),
        ),
        termination=mapped_termination(case.termination),
        search_budget=case.search_budget,
        context_budget=case.context_budget,
        intervention=case.intervention,
    )
