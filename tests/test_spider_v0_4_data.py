from __future__ import annotations

from collections import Counter

import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    V04_DATASET_VERSION,
    FreshRenderedBatchSource,
    SyntheticManifoldRenderer,
    build_aligned_dev_manifest,
    default_aligned_dev_specs,
    generate_aligned_dev_cases,
    pack_rendered_cases,
    verify_case,
)
from hippocampus.programs.schema import ProgramFamily, TerminationDecision
from hippocampus.spider import (
    SpiderModel,
    SpiderModelConfig,
    TrainingLoopConfig,
    train_oracle_batches,
)


def test_v0_4_partitions_are_disjoint_and_stratified() -> None:
    specs = default_aligned_dev_specs()

    assert [spec.name for spec in specs] == [
        "training",
        "model_selection",
        "calibration",
        "development_evaluation",
    ]
    assert [spec.case_count for spec in specs] == [8192, 512, 512, 1024]
    assert all(spec.dataset_version == V04_DATASET_VERSION for spec in specs)
    assert not any(spec.sealed for spec in specs)

    all_case_ids: set[str] = set()
    for spec in specs:
        cases = generate_aligned_dev_cases(spec, limit=256)
        family_counts = Counter(case.family.value for case in cases)
        outcome_counts = Counter(case.answerable for case in cases)
        graph_size_counts = Counter(len(case.nodes) for case in cases)

        assert set(family_counts.values()) == {64}
        assert outcome_counts == {True: 128, False: 128}
        assert set(graph_size_counts) == {8, 16, 24, 32}
        assert set(graph_size_counts.values()) == {64}
        assert all_case_ids.isdisjoint(case.case_id for case in cases)
        all_case_ids.update(case.case_id for case in cases)


def test_v0_4_manifest_hashes_actual_cases_deterministically() -> None:
    spec = default_aligned_dev_specs()[1]
    first_cases = generate_aligned_dev_cases(spec, limit=128)
    second_cases = generate_aligned_dev_cases(spec, limit=128)
    first = build_aligned_dev_manifest(spec, first_cases)
    second = build_aligned_dev_manifest(spec, second_cases)

    assert first == second
    assert len(first.sha256) == 64
    assert first.case_ids == tuple(case.case_id for case in first_cases)
    assert first.base_case_ids == tuple(
        case.base_case_id for case in first_cases
    )
    assert first.distributions["family"] == {
        "corroboration": 32,
        "latest_valid": 32,
        "lookup": 32,
        "reachability": 32,
    }


def test_unsupported_latest_valid_case_does_not_require_context_work() -> None:
    spec = default_aligned_dev_specs()[0]
    case = next(
        case
        for case in generate_aligned_dev_cases(spec, limit=256)
        if case.family is ProgramFamily.LATEST_VALID
        and case.termination.decision is TerminationDecision.UNKNOWN_UNSUPPORTED
    )

    assert verify_case(case).valid
    assert not any(
        candidate.context_has_value
        for round_ in case.trace.rounds
        for candidate in round_.candidates
    )


def _fresh_source_fixture():
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    spec = default_aligned_dev_specs()[0]
    cases = generate_aligned_dev_cases(spec, limit=8)
    renderer = SyntheticManifoldRenderer(
        schema,
        query_dim=8,
        seed=401,
        geometry="orthogonal_aligned",
    )
    source = FreshRenderedBatchSource(
        cases,
        renderer=renderer,
        schema=schema,
        base_row_seed=1701,
    )
    return schema, cases, renderer, source


def test_fresh_batch_source_rerenders_and_restores_presentation_state() -> None:
    _, cases, renderer, source = _fresh_source_fixture()
    presentations = [source[0] for _ in range(8)]

    assert source.case_ids == tuple(case.case_id for case in cases)
    assert source.presentation_counts[0] == 8
    first_values = presentations[0].graph.summaries.values
    assert any(
        not torch.equal(first_values, item.graph.summaries.values)
        for item in presentations[1:]
    )
    assert torch.equal(
        first_values.sort(dim=0).values,
        presentations[-1].graph.summaries.values.sort(dim=0).values,
    )

    state = source.state_dict()
    expected_next = source[3]
    restored = FreshRenderedBatchSource(
        cases,
        renderer=renderer,
        schema=source.schema,
        base_row_seed=1701,
    )
    restored.load_state_dict(state)
    actual_next = restored[3]
    assert torch.equal(
        expected_next.graph.summaries.values,
        actual_next.graph.summaries.values,
    )


def test_training_resume_is_exact_with_fresh_presentations(tmp_path) -> None:
    schema, cases, renderer, full_source = _fresh_source_fixture()
    monitor = tuple(
        pack_rendered_cases(
            (case,),
            (renderer.render(case, row_permutation_seed=90_000 + index),),
            schema=schema,
        )
        for index, case in enumerate(cases[:2])
    )
    config = SpiderModelConfig(
        summary_dim=8,
        context_dim=8,
        edge_dim=8,
        query_dim=8,
        d_model=16,
        num_heads=4,
        num_blocks=1,
        path_rows=3,
        evidence_rows=3,
        dropout=0.0,
    )
    loop = TrainingLoopConfig(
        steps=4,
        batch_size=2,
        learning_rate=0.001,
        seed=71,
        log_every=2,
    )
    torch.manual_seed(991)
    full_model = SpiderModel(config)
    initial = {
        name: value.detach().clone()
        for name, value in full_model.state_dict().items()
    }
    initial_rng = torch.get_rng_state()
    torch.set_rng_state(initial_rng)
    train_oracle_batches(
        full_model,
        full_source,
        loop_config=loop,
        monitor_batches=monitor,
    )

    paused_source = FreshRenderedBatchSource(
        cases,
        renderer=renderer,
        schema=schema,
        base_row_seed=1701,
    )
    paused_model = SpiderModel(config)
    paused_model.load_state_dict(initial)
    checkpoint = tmp_path / "fresh.pt"
    torch.set_rng_state(initial_rng)
    train_oracle_batches(
        paused_model,
        paused_source,
        loop_config=loop,
        monitor_batches=monitor,
        checkpoint_path=checkpoint,
        stop_after_steps=2,
    )

    resumed_source = FreshRenderedBatchSource(
        cases,
        renderer=renderer,
        schema=schema,
        base_row_seed=1701,
    )
    resumed_model = SpiderModel(config)
    train_oracle_batches(
        resumed_model,
        resumed_source,
        loop_config=loop,
        monitor_batches=monitor,
        checkpoint_path=checkpoint,
        resume_checkpoint=checkpoint,
    )

    for name, expected in full_model.state_dict().items():
        assert torch.equal(expected, resumed_model.state_dict()[name]), name
    assert full_source.presentation_counts == resumed_source.presentation_counts
