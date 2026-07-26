from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    CounterfactualKind,
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    TerminationDecision,
    build_split_manifest,
    default_split_specs,
    generate_split_cases,
    make_counterfactual,
    make_equivalent_view,
    metadata_leakage_report,
    pack_rendered_cases,
    swap_aligned_edge_manifolds,
    verify_case,
)
from hippocampus.programs.schema import ObservableAtom


@pytest.fixture
def generator() -> GraphProgramGenerator:
    return GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=8,
            max_nodes=14,
            min_path_length=1,
            max_path_length=4,
            min_summary_rows=1,
            max_summary_rows=4,
            min_context_rows=0,
            max_context_rows=5,
        )
    )


@pytest.mark.parametrize("family", tuple(ProgramFamily))
def test_generator_is_deterministic_and_oracle_is_valid(
    generator: GraphProgramGenerator,
    family: ProgramFamily,
) -> None:
    first = generator.generate(family=family, seed=901)
    second = generator.generate(family=family, seed=901)

    assert first == second
    assert first.family is family
    assert first.case_id == second.case_id
    assert verify_case(first).valid
    assert first.trace.rounds
    assert len(first.nodes) >= 8


def test_all_families_generate_answerable_and_unknown_cases(
    generator: GraphProgramGenerator,
) -> None:
    for family in ProgramFamily:
        answerable = generator.generate(family=family, seed=1201, answerable=True)
        unknown = generator.generate(family=family, seed=1202, answerable=False)

        assert answerable.termination.decision is TerminationDecision.ANSWER
        assert answerable.answer_nodes
        assert unknown.termination.decision is not TerminationDecision.ANSWER
        assert not unknown.answer_nodes
        assert verify_case(answerable).valid
        assert verify_case(unknown).valid


@pytest.mark.parametrize(
    "decision",
    [
        TerminationDecision.UNKNOWN_INCOMPLETE,
        TerminationDecision.UNKNOWN_UNSUPPORTED,
    ],
)
def test_generic_budget_and_unsupported_unknown_cases(
    generator: GraphProgramGenerator,
    decision: TerminationDecision,
) -> None:
    case = generator.generate(
        family=ProgramFamily.REACHABILITY,
        seed=1407,
        unknown_decision=decision,
    )

    assert case.termination.decision is decision
    assert not case.answer_nodes
    assert not case.evidence_nodes
    if decision is TerminationDecision.UNKNOWN_INCOMPLETE:
        assert case.search_budget == 0
    else:
        assert case.search_budget > 0
        assert len(case.query_atoms) >= 5
    assert verify_case(case).valid


def test_context_budget_exhaustion_is_explicit(
    generator: GraphProgramGenerator,
) -> None:
    case = generator.generate(
        family=ProgramFamily.LATEST_VALID,
        seed=880,
        unknown_decision=TerminationDecision.UNKNOWN_INCOMPLETE,
        context_budget_exhausted=True,
    )

    assert case.context_budget == 0
    assert case.search_budget > 0
    assert any(
        candidate.context_has_value
        for round_ in case.trace.rounds
        for candidate in round_.candidates
    )
    assert verify_case(case).valid


def test_parallel_trace_retains_multiple_acceptable_paths(
    generator: GraphProgramGenerator,
) -> None:
    case = generator.generate(
        family=ProgramFamily.REACHABILITY,
        seed=414,
        answerable=True,
        require_multiple_paths=True,
    )

    assert len(case.trace.valid_paths) >= 2
    assert any(
        sum(candidate.acceptable for candidate in round_.candidates) >= 2
        for round_ in case.trace.rounds
    )
    assert verify_case(case).valid


def test_latest_valid_requires_context_information(
    generator: GraphProgramGenerator,
) -> None:
    case = generator.generate(
        family=ProgramFamily.LATEST_VALID,
        seed=2307,
        answerable=True,
    )

    positive_reads = [
        candidate
        for round_ in case.trace.rounds
        for candidate in round_.candidates
        if candidate.context_has_value
    ]
    assert positive_reads
    for candidate in positive_reads:
        node = case.nodes[candidate.destination_node]
        assert node.context_atoms
        assert set(node.context_atoms) - set(node.summary_atoms)
    assert verify_case(case).valid


def test_observable_schema_has_no_supervisor_or_position_fields() -> None:
    names = {field.name for field in fields(ObservableAtom)}
    forbidden = {
        "answer",
        "answerable",
        "family",
        "node_id",
        "edge_id",
        "relation_id",
        "row_index",
        "position",
        "path",
        "provenance",
        "target",
    }
    assert names.isdisjoint(forbidden)


def test_equivalent_view_changes_surface_and_preserves_truth(
    generator: GraphProgramGenerator,
) -> None:
    case = generator.generate(
        family=ProgramFamily.REACHABILITY,
        seed=88,
        answerable=True,
    )
    view = make_equivalent_view(case, seed=99)

    assert view.case_id != case.case_id
    assert view.base_case_id == case.base_case_id
    assert view.query_atoms != case.query_atoms
    assert view.answer_latent_ids == case.answer_latent_ids
    assert view.evidence_latent_ids == case.evidence_latent_ids
    assert view.latent_edge_ids == case.latent_edge_ids
    assert verify_case(view).valid


def test_functional_edge_swapping_uses_latent_alignment(
    generator: GraphProgramGenerator,
) -> None:
    case = generator.generate(
        family=ProgramFamily.REACHABILITY,
        seed=188,
        answerable=True,
    )
    view = make_equivalent_view(case, seed=199)
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=12)
    receiving = renderer.render(case)
    donor = renderer.render(view)
    swap = swap_aligned_edge_manifolds(
        case,
        receiving,
        view,
        donor,
    )
    donor_by_latent = {
        edge.latent_id: edge_id for edge_id, edge in enumerate(view.edges)
    }

    assert swap.rendered.case_id == case.case_id
    assert swap.rendered.query is receiving.query
    assert all(
        swapped is original
        for swapped, original in zip(
            swap.rendered.summaries,
            receiving.summaries,
            strict=True,
        )
    )
    assert any(
        not torch.equal(original, swapped)
        for original, swapped in zip(
            receiving.edges,
            swap.rendered.edges,
            strict=True,
        )
    )
    for latent_id, receiving_id in zip(
        swap.latent_edge_ids,
        swap.receiving_edge_ids,
        strict=True,
    ):
        assert torch.equal(
            swap.rendered.edges[receiving_id],
            donor.edges[donor_by_latent[latent_id]],
        )
    assert case.answer_latent_ids == view.answer_latent_ids


def test_one_edit_counterfactual_recomputes_truth(
    generator: GraphProgramGenerator,
) -> None:
    case = generator.generate(
        family=ProgramFamily.LOOKUP,
        seed=340,
        answerable=True,
    )
    edited = make_counterfactual(
        case,
        kind=CounterfactualKind.REMOVE_DECISIVE_EDGE,
    )

    assert edited.base_case_id == case.base_case_id
    assert len(edited.edges) == len(case.edges) - 1
    assert edited.nodes == case.nodes
    assert edited.query_atoms == case.query_atoms
    assert edited.termination.decision is TerminationDecision.UNKNOWN_ABSENT
    assert not edited.answer_nodes
    assert verify_case(edited).valid


@pytest.mark.parametrize(
    "kind",
    [
        CounterfactualKind.REVERSE_DECISIVE_EDGE,
        CounterfactualKind.REPLACE_ENDPOINT,
        CounterfactualKind.DISCONNECT_ONLY_PATH,
    ],
)
def test_structural_counterfactual_variants_are_exact(
    generator: GraphProgramGenerator,
    kind: CounterfactualKind,
) -> None:
    case = generator.generate(
        family=ProgramFamily.LOOKUP,
        seed=910,
        answerable=True,
    )
    edited = make_counterfactual(case, kind=kind)

    assert edited.termination.decision is TerminationDecision.UNKNOWN_ABSENT
    assert edited.intervention is not None
    assert edited.intervention.kind is kind
    assert edited.nodes == case.nodes
    assert edited.query_atoms == case.query_atoms
    assert verify_case(edited).valid


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (
            CounterfactualKind.ALTER_TEMPORAL_VALUE,
            TerminationDecision.ANSWER,
        ),
        (
            CounterfactualKind.ADD_NEWER_CONFLICT,
            TerminationDecision.UNKNOWN_CONFLICT,
        ),
    ],
)
def test_temporal_counterfactual_changes_one_context_observation(
    generator: GraphProgramGenerator,
    kind: CounterfactualKind,
    expected: TerminationDecision,
) -> None:
    case = generator.generate(
        family=ProgramFamily.LATEST_VALID,
        seed=912,
        answerable=True,
    )
    edited = make_counterfactual(case, kind=kind)
    changed_rows = sum(
        first.context_atoms != second.context_atoms
        for first, second in zip(case.nodes, edited.nodes, strict=True)
    )

    assert changed_rows == 1
    assert edited.termination.decision is expected
    if expected is TerminationDecision.ANSWER:
        assert edited.answer_latent_ids != case.answer_latent_ids
    else:
        assert len(edited.evidence_nodes) == 2
    assert verify_case(edited).valid


def test_source_invalidation_changes_one_context_and_removes_corroboration(
    generator: GraphProgramGenerator,
) -> None:
    case = generator.generate(
        family=ProgramFamily.CORROBORATION,
        seed=914,
        answerable=True,
    )
    edited = make_counterfactual(
        case,
        kind=CounterfactualKind.INVALIDATE_SOURCE,
    )
    changed_rows = sum(
        first.context_atoms != second.context_atoms
        for first, second in zip(case.nodes, edited.nodes, strict=True)
    )

    assert changed_rows == 1
    assert edited.termination.decision is TerminationDecision.UNKNOWN_ABSENT
    assert len(edited.evidence_nodes) == 1
    assert verify_case(edited).valid


def _sorted_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] < 2:
        return tensor
    weights = torch.linspace(1.0, 2.0, tensor.shape[1], dtype=tensor.dtype)
    order = torch.argsort(tensor @ weights, stable=True)
    return tensor[order]


def test_renderer_row_permutations_preserve_multisets(
    generator: GraphProgramGenerator,
) -> None:
    case = generator.generate(
        family=ProgramFamily.CORROBORATION,
        seed=771,
        answerable=True,
    )
    renderer = SyntheticManifoldRenderer(
        GraphSchema(summary_dim=16, context_dim=16, edge_dim=16),
        query_dim=16,
        seed=7,
    )
    first = renderer.render(case, row_permutation_seed=1)
    second = renderer.render(case, row_permutation_seed=2)

    assert torch.allclose(_sorted_rows(first.query), _sorted_rows(second.query))
    for family_a, family_b in (
        (first.summaries, second.summaries),
        (first.contexts, second.contexts),
        (first.edges, second.edges),
    ):
        assert len(family_a) == len(family_b)
        for rows_a, rows_b in zip(family_a, family_b, strict=True):
            assert torch.allclose(_sorted_rows(rows_a), _sorted_rows(rows_b))


def test_renderer_variable_cardinality_and_packed_batch(
    generator: GraphProgramGenerator,
) -> None:
    cases = [
        generator.generate(family=family, seed=500 + index)
        for index, family in enumerate(ProgramFamily)
    ]
    schema = GraphSchema(summary_dim=12, context_dim=10, edge_dim=14)
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=31)
    rendered = [
        renderer.render(case, row_permutation_seed=100 + index)
        for index, case in enumerate(cases)
    ]
    packed = pack_rendered_cases(cases, rendered, schema=schema)

    assert packed.graph.validate() is packed.graph
    assert packed.query.owner_count == len(cases)
    assert packed.query.width == 8
    assert packed.graph.topology.graph_count == len(cases)
    assert packed.graph.summaries.owner_count == sum(len(case.nodes) for case in cases)
    assert packed.graph.edges.owner_count == sum(len(case.edges) for case in cases)
    assert len(set(packed.graph.summaries.lengths.tolist())) > 1
    assert any(length == 0 for length in packed.graph.contexts.lengths.tolist())


def test_split_manifests_are_disjoint_and_hashed() -> None:
    specs = default_split_specs(case_scale=0.02)
    manifests = [build_split_manifest(spec) for spec in specs]
    all_ids: set[str] = set()

    for manifest in manifests:
        ids = set(manifest.case_ids)
        assert ids
        assert len(manifest.sha256) == 64
        assert all_ids.isdisjoint(ids)
        all_ids.update(ids)


def test_training_split_contains_all_selective_outcome_classes() -> None:
    train = next(
        spec for spec in default_split_specs() if spec.name == "train"
    )
    cases = generate_split_cases(train, limit=24)
    decisions = {case.termination.decision for case in cases}

    assert TerminationDecision.ANSWER in decisions
    assert TerminationDecision.UNKNOWN_ABSENT in decisions
    assert TerminationDecision.UNKNOWN_CONFLICT in decisions
    assert TerminationDecision.UNKNOWN_INCOMPLETE in decisions
    assert TerminationDecision.UNKNOWN_UNSUPPORTED in decisions


def test_metadata_leakage_stays_near_majority_baseline(
    generator: GraphProgramGenerator,
) -> None:
    cases = generator.generate_suite(cases_per_family=24, seed=2901)
    report = metadata_leakage_report(cases, folds=4)

    assert report.case_count == 96
    assert report.answerability_advantage <= 0.10
    assert report.fixed_answer_position_rate <= 0.20
    assert report.fixed_edge_position_rate <= 0.20
