from __future__ import annotations

import torch

from hippocampus import FrontierExpansion, GraphSchema
from hippocampus.programs import (
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from hippocampus.spider import (
    CandidateOutputs,
    ControllerState,
    HypothesisBatch,
    SparseControllerConfig,
    SparseWavefrontController,
    SpiderModel,
    SpiderModelConfig,
    stable_candidate_selection,
)


def _expansion(
    arc_ids: list[int],
    destinations: list[int],
    frontier_positions: list[int] | None = None,
) -> FrontierExpansion:
    count = len(arc_ids)
    positions = frontier_positions or list(range(count))
    return FrontierExpansion(
        arc_ids=torch.tensor(arc_ids, dtype=torch.int32),
        edge_ids=torch.arange(count, dtype=torch.int32),
        source_node_ids=torch.zeros(count, dtype=torch.int32),
        destination_node_ids=torch.tensor(destinations, dtype=torch.int32),
        frontier_positions=torch.tensor(positions, dtype=torch.int64),
        arc_offsets=torch.tensor([0, count], dtype=torch.int32),
    )


def test_stable_topk_uses_arc_id_and_destination_caps() -> None:
    expansion = _expansion([3, 1, 2, 0], [5, 5, 6, 6])
    selected = stable_candidate_selection(
        expansion,
        torch.zeros(4),
        frontier_width=2,
        hypotheses_per_node=1,
    )

    assert selected.tolist() == [3, 1]


def test_duplicate_frontier_occurrences_remain_distinct() -> None:
    expansion = _expansion([2, 2], [4, 4], [0, 1])
    selected = stable_candidate_selection(
        expansion,
        torch.ones(2),
        frontier_width=2,
        hypotheses_per_node=2,
    )

    assert selected.tolist() == [0, 1]


def _batch_and_model(family: ProgramFamily = ProgramFamily.LOOKUP):
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    generator = GraphProgramGenerator(
        GeneratorConfig(min_nodes=8, max_nodes=8)
    )
    case = generator.generate(family=family, seed=82, answerable=True)
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=9)
    rendered = renderer.render(case)
    batch = pack_rendered_cases((case,), (rendered,), schema=schema)
    model = SpiderModel(
        SpiderModelConfig(
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
    ).eval()
    return case, batch, model


def test_controller_step_uses_real_frontier_expansion_and_duplicates() -> None:
    _, batch, model = _batch_and_model()
    initial = model.initial_hypotheses(batch)
    hypotheses = initial.repeat_occurrences(torch.tensor([0, 0]))
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=2,
            frontier_width=8,
            hypotheses_per_node=2,
            context_read_budget=0,
            search_budget=100,
            max_depth=4,
        )
    )
    state = ControllerState.initial()

    step = controller.step(
        model,
        batch,
        hypotheses,
        model.initial_evidence(batch),
        state,
    )

    assert step.expansion.arc_offsets.numel() == 3
    assert step.expansion.total_arcs % 2 == 0
    half = step.expansion.total_arcs // 2
    assert torch.equal(
        step.expansion.arc_ids[:half],
        step.expansion.arc_ids[half:],
    )
    assert set(step.expansion.frontier_positions.tolist()) == {0, 1}


def test_isolated_frontier_and_exhausted_budget_are_canonical() -> None:
    _, batch, model = _batch_and_model()
    topology = batch.graph.topology
    isolated = next(
        node
        for node in range(topology.node_count)
        if topology.adjacency_row_ptr[node] == topology.adjacency_row_ptr[node + 1]
    )
    template = model.initial_hypotheses(batch)
    hypotheses = HypothesisBatch(
        node_ids=torch.tensor([isolated], dtype=torch.int32),
        graph_ids=torch.zeros(1, dtype=torch.int32),
        path_state=template.path_state,
        scores=torch.zeros(1),
        depths=torch.zeros(1, dtype=torch.int32),
        parent_trace_ids=torch.full((1,), -1, dtype=torch.int64),
        incoming_arc_ids=torch.full((1,), -1, dtype=torch.int32),
        incoming_edge_ids=torch.full((1,), -1, dtype=torch.int32),
        context_read=torch.zeros(1, dtype=torch.bool),
    )
    controller = SparseWavefrontController(
        SparseControllerConfig(search_budget=0, context_read_budget=0)
    )

    step = controller.step(
        model,
        batch,
        hypotheses,
        model.initial_evidence(batch),
        ControllerState.initial(),
    )

    assert step.expansion.total_arcs == 0
    assert step.expansion.arc_offsets.tolist() == [0, 0]
    assert step.next_hypotheses.count == 0
    assert step.state.search_budget_exhausted


def test_cycle_execution_is_bounded_and_replay_is_deterministic() -> None:
    _, batch, model = _batch_and_model(ProgramFamily.REACHABILITY)
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=3,
            frontier_width=4,
            hypotheses_per_node=2,
            context_read_budget=1,
            search_budget=100,
            max_depth=3,
        )
    )

    first = controller.run(model, batch)
    second = controller.run(model, batch)

    assert first.rounds <= 3
    assert first.selected_arc_trace == second.selected_arc_trace
    assert first.termination == second.termination


def test_exact_evidence_ledger_records_selected_references() -> None:
    _, batch, model = _batch_and_model()
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=1,
            frontier_width=8,
            hypotheses_per_node=2,
            context_read_budget=2,
            search_budget=100,
            evidence_threshold=0.0,
        )
    )
    result = controller.run(model, batch)

    assert result.evidence_ledger
    for entry in result.evidence_ledger:
        assert entry.arc_id >= 0
        assert entry.edge_id >= 0
        assert entry.node_id >= 0
        assert entry.round_index == 0
