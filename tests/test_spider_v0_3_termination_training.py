from __future__ import annotations

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
    ActionSchedule,
    SparseControllerConfig,
    SpiderLossConfig,
    SpiderModel,
    SpiderModelConfig,
    collect_termination_state_dataset,
    evaluate_termination_state_dataset,
    train_frozen_null_head,
    train_frozen_termination_head,
)


def _fixture(*, use_null: bool = False):
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    generator = GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=8,
            max_nodes=10,
            min_path_length=2,
            max_path_length=3,
        )
    )
    cases = (
        generator.generate(
            family=ProgramFamily.REACHABILITY,
            seed=41,
            answerable=True,
            require_multiple_paths=True,
        ),
        generator.generate(
            family=ProgramFamily.LOOKUP,
            seed=42,
            answerable=False,
        ),
    )
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=43)
    batches = tuple(
        pack_rendered_cases(
            (case,),
            (renderer.render(case),),
            schema=schema,
        )
        for case in cases
    )
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
            termination_mode="factorized",
            use_null_expansion=use_null,
        )
    )
    controller = SparseControllerConfig(
        max_rounds=4,
        frontier_width=6,
        hypotheses_per_node=2,
        context_read_budget=4,
        evidence_selection_budget=4,
        search_budget=64,
        max_depth=5,
        expansion_policy="threshold",
    )
    return model, batches, controller


def test_state_collection_uses_direct_factors_and_detached_inputs() -> None:
    model, batches, controller = _fixture()
    dataset = collect_termination_state_dataset(
        model,
        batches,
        controller_config=controller,
        schedules=(
            ActionSchedule.oracle_only(),
            ActionSchedule.model_only(),
        ),
        seed=7,
    )

    assert dataset.count >= len(batches)
    assert dataset.query.shape == (dataset.count, model.config.d_model)
    assert dataset.control.shape == (
        dataset.count,
        model.config.control_width,
    )
    assert not dataset.query.requires_grad
    assert dataset.factor_targets.evidence_sufficient.dtype == torch.bool
    assert dataset.factor_targets.useful_work_remaining.any()
    assert dataset.case_ids[0]
    assert len(dataset.round_indices) == dataset.count


def test_balanced_frozen_termination_training_changes_only_terminator() -> None:
    torch.manual_seed(51)
    model, batches, controller = _fixture()
    dataset = collect_termination_state_dataset(
        model,
        batches,
        controller_config=controller,
        schedules=(
            ActionSchedule.oracle_only(),
            ActionSchedule.model_only(),
        ),
        seed=9,
    )
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    initial = evaluate_termination_state_dataset(
        model,
        dataset,
        loss_config=SpiderLossConfig(),
    )

    result = train_frozen_termination_head(
        model,
        dataset,
        steps=40,
        batch_size=8,
        learning_rate=0.01,
        seed=10,
        loss_config=SpiderLossConfig(),
    )
    final = evaluate_termination_state_dataset(
        model,
        dataset,
        loss_config=SpiderLossConfig(),
    )

    assert result.completed_steps == 40
    assert final.loss < initial.loss
    changed_termination = False
    for name, value in model.state_dict().items():
        if name.startswith("termination_head."):
            changed_termination |= not torch.equal(value, before[name])
        else:
            assert torch.equal(value, before[name]), name
    assert changed_termination


def test_null_training_changes_only_branch_null_head() -> None:
    torch.manual_seed(61)
    model, batches, controller = _fixture(use_null=True)
    dataset = collect_termination_state_dataset(
        model,
        batches,
        controller_config=controller,
        schedules=(ActionSchedule.oracle_only(),),
        seed=11,
        collect_null_states=True,
    )
    assert dataset.null_states is not None
    assert dataset.null_states.count > 0
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    result = train_frozen_null_head(
        model,
        dataset.null_states,
        steps=20,
        batch_size=8,
        learning_rate=0.01,
        seed=12,
    )

    assert result.completed_steps == 20
    changed_null = False
    for name, value in model.state_dict().items():
        if name.startswith("null_expansion_head."):
            changed_null |= not torch.equal(value, before[name])
        else:
            assert torch.equal(value, before[name]), name
    assert changed_null
