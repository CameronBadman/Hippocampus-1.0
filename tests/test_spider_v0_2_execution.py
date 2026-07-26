from __future__ import annotations

from dataclasses import replace

import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ControllerExecutionPolicy,
    PathStateIntervention,
    SparseControllerConfig,
    SparseWavefrontController,
    SpiderModel,
    SpiderModelConfig,
    apply_path_state_intervention,
)
from hippocampus.spider.hypothesis import HypothesisBatch
from hippocampus.spider.terminator import TerminationOutput


class _AlwaysAnswerSpider(SpiderModel):
    def __init__(self, config: SpiderModelConfig) -> None:
        super().__init__(config)
        self.termination_calls = 0

    def termination_output(
        self,
        batch,
        hypotheses,
        evidence,
        controller_features=None,
    ) -> TerminationOutput:
        del hypotheses, controller_features
        self.termination_calls += 1
        logits = evidence.new_full((batch.graph_count, 6), -20.0)
        logits[:, 1] = 20.0
        return TerminationOutput(logits=logits)


def _fixture(path_length: int = 4):
    torch.manual_seed(17)
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    case = GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=12,
            max_nodes=12,
            min_path_length=path_length,
            max_path_length=path_length,
        )
    ).generate(
        family=ProgramFamily.REACHABILITY,
        seed=824,
        answerable=True,
        require_multiple_paths=True,
    )
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=91)
    batch = pack_rendered_cases(
        (case,),
        (renderer.render(case),),
        schema=schema,
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
    )
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=8,
            frontier_width=8,
            hypotheses_per_node=2,
            context_read_budget=4,
            evidence_selection_budget=8,
            search_budget=256,
            max_depth=10,
            expand_threshold=0.0,
        )
    )
    return case, batch, config, controller


def test_fixed_horizon_ignores_intermediate_learned_stopping() -> None:
    _, batch, model_config, controller = _fixture()
    model = _AlwaysAnswerSpider(model_config).eval()

    learned = controller.run(model, batch)
    assert learned.rounds == 1
    assert model.termination_calls == 1

    model.termination_calls = 0
    fixed = controller.run(
        model,
        batch,
        execution_policy=ControllerExecutionPolicy.fixed(4),
    )
    assert fixed.rounds == 4
    assert model.termination_calls == 1
    assert len(fixed.action_diagnostics) == 4


def test_oracle_required_horizon_is_resolved_without_model_features() -> None:
    case, batch, model_config, controller = _fixture(path_length=4)
    model = _AlwaysAnswerSpider(model_config).eval()
    result = controller.run(
        model,
        batch,
        execution_policy=ControllerExecutionPolicy.oracle_required(),
    )

    assert result.rounds == len(case.trace.rounds)
    assert model.termination_calls == 1


def test_reset_rebuilds_query_conditioned_initial_state() -> None:
    _, batch, model_config, _ = _fixture()
    model = SpiderModel(model_config).eval()
    initial = model.initial_hypotheses(batch)
    changed = replace(initial, path_state=initial.path_state + 7.0)
    reset = apply_path_state_intervention(
        model,
        batch,
        changed,
        intervention=PathStateIntervention.RESET,
        round_index=1,
        seed=4,
    )

    expected = model.initial_path_state(batch, changed.graph_ids)
    assert torch.equal(reset.path_state, expected)
    assert torch.equal(reset.node_ids, changed.node_ids)


def test_detach_preserves_values_and_cuts_cross_round_gradient() -> None:
    _, batch, model_config, _ = _fixture()
    model = SpiderModel(model_config)
    initial = model.initial_hypotheses(batch)
    detached = apply_path_state_intervention(
        model,
        batch,
        initial,
        intervention=PathStateIntervention.DETACH,
        round_index=1,
        seed=4,
    )

    assert torch.equal(detached.path_state, initial.path_state)
    assert detached.path_state.grad_fn is None
    assert not detached.path_state.requires_grad


def test_shuffle_is_seeded_graph_local_and_preserves_state_multiset() -> None:
    _, batch, model_config, _ = _fixture()
    model = SpiderModel(model_config).eval()
    initial = model.initial_hypotheses(batch)
    repeated = initial.repeat_occurrences(torch.tensor([0, 0, 0, 0]))
    path = torch.arange(
        repeated.path_state.numel(),
        dtype=repeated.path_state.dtype,
    ).reshape_as(repeated.path_state)
    repeated = replace(repeated, path_state=path)

    first = apply_path_state_intervention(
        model,
        batch,
        repeated,
        intervention=PathStateIntervention.SHUFFLE,
        round_index=2,
        seed=77,
    )
    second = apply_path_state_intervention(
        model,
        batch,
        repeated,
        intervention=PathStateIntervention.SHUFFLE,
        round_index=2,
        seed=77,
    )

    assert torch.equal(first.path_state, second.path_state)
    assert torch.equal(
        first.path_state.flatten(1).sort(dim=0).values,
        repeated.path_state.flatten(1).sort(dim=0).values,
    )
    assert torch.equal(first.node_ids, repeated.node_ids)


def test_pooled_current_node_replacement_ignores_previous_path_state() -> None:
    _, batch, model_config, _ = _fixture()
    model = SpiderModel(model_config).eval()
    initial = model.initial_hypotheses(batch)
    first = apply_path_state_intervention(
        model,
        batch,
        initial,
        intervention=PathStateIntervention.POOLED_CURRENT_NODE,
        round_index=1,
        seed=0,
    )
    changed = replace(initial, path_state=initial.path_state + 100.0)
    second = apply_path_state_intervention(
        model,
        batch,
        changed,
        intervention=PathStateIntervention.POOLED_CURRENT_NODE,
        round_index=1,
        seed=0,
    )

    assert torch.equal(first.path_state, second.path_state)
    assert first.path_state.shape == initial.path_state.shape


def test_intervention_keeps_all_hypothesis_metadata() -> None:
    _, batch, model_config, _ = _fixture()
    model = SpiderModel(model_config).eval()
    initial = model.initial_hypotheses(batch)
    intervened = apply_path_state_intervention(
        model,
        batch,
        initial,
        intervention=PathStateIntervention.RESET,
        round_index=1,
        seed=0,
    )

    for field in HypothesisBatch.__dataclass_fields__:
        if field != "path_state":
            assert torch.equal(getattr(intervened, field), getattr(initial, field))
