from __future__ import annotations

from collections import Counter

from hippocampus.programs import (
    V07_DATASET_VERSION,
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    audit_aligned_program_labels,
    default_binding_specs,
    generate_binding_cases,
    observable_symbols,
    verify_case,
)


def _generator() -> GraphProgramGenerator:
    return GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=8,
            max_nodes=8,
            min_path_length=1,
            max_path_length=1,
            generator_version=V07_DATASET_VERSION,
            lookup_binding_mode="matched_conjunction",
        )
    )


def _outgoing(case):
    start = case.start_nodes[0]
    return tuple(edge for edge in case.edges if edge.source_node == start)


def _scalar_inventory(case) -> Counter[float]:
    return Counter(
        atom.scalar
        for edge in case.edges
        for atom in edge.atoms
        if atom.scalar is not None
    )


def test_matched_lookup_requires_relation_value_and_validity() -> None:
    case = _generator().generate(
        family=ProgramFamily.LOOKUP,
        seed=71,
        answerable=True,
    )
    query_symbols = {
        symbol for atom in case.query_atoms for symbol in atom.symbols
    }
    candidates = _outgoing(case)

    assert len(candidates) == 4
    assert {len(edge.atoms) for edge in candidates} == {2}
    assert (
        len(
            {
                len(case.nodes[edge.destination_node].summary_atoms)
                for edge in candidates
            }
        )
        == 1
    )

    patterns: Counter[tuple[bool, bool, bool]] = Counter()
    for edge in candidates:
        edge_symbols = {
            symbol for atom in edge.atoms for symbol in atom.symbols
        }
        destination_symbols = {
            symbol
            for atom in case.nodes[edge.destination_node].summary_atoms
            for symbol in atom.symbols
        }
        patterns[
            (
                bool(query_symbols & edge_symbols),
                bool(query_symbols & destination_symbols),
                edge.valid,
            )
        ] += 1

    assert patterns == Counter(
        {
            (True, True, True): 1,
            (True, False, True): 1,
            (False, True, True): 1,
            (True, True, False): 1,
        }
    )
    assert sum(
        target.include_as_evidence
        for target in case.trace.rounds[0].candidates
    ) == 1
    assert verify_case(case).valid


def test_lookup_negative_is_metadata_matched_and_breaks_the_conjunction() -> None:
    generator = _generator()
    positive = generator.generate(
        family=ProgramFamily.LOOKUP,
        seed=73,
        answerable=True,
    )
    negative = generator.generate(
        family=ProgramFamily.LOOKUP,
        seed=73,
        answerable=False,
    )

    assert positive.query_atoms == negative.query_atoms
    assert observable_symbols((positive,)) == observable_symbols((negative,))
    assert _scalar_inventory(positive) == _scalar_inventory(negative)
    assert [len(node.summary_atoms) for node in positive.nodes] == [
        len(node.summary_atoms) for node in negative.nodes
    ]
    assert [
        (edge.source_node, edge.destination_node, len(edge.atoms))
        for edge in positive.edges
    ] == [
        (edge.source_node, edge.destination_node, len(edge.atoms))
        for edge in negative.edges
    ]
    assert all(
        not target.include_as_evidence
        for target in negative.trace.rounds[0].candidates
    )
    assert verify_case(positive).valid
    assert verify_case(negative).valid


def test_v0_7_partitions_are_versioned_and_symbol_disjoint() -> None:
    specs = default_binding_specs()

    assert [spec.case_count for spec in specs] == [8192, 512, 512, 1024]
    assert all(spec.dataset_version == V07_DATASET_VERSION for spec in specs)
    assert all(not spec.sealed for spec in specs)

    seen: set[str] = set()
    for spec in specs:
        cases = generate_binding_cases(spec, limit=128)
        symbols = observable_symbols(cases)
        assert symbols
        assert seen.isdisjoint(symbols)
        seen.update(symbols)

        lookup = [case for case in cases if case.family is ProgramFamily.LOOKUP]
        assert lookup
        audit = audit_aligned_program_labels(cases)
        assert audit.invalid_case_count == 0
        assert audit.evidence_label_mismatch_count == 0
