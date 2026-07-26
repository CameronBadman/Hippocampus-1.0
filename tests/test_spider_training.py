from __future__ import annotations

import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    GeneratorConfig,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from hippocampus.spider import (
    SpiderLossConfig,
    SpiderModel,
    SpiderModelConfig,
    TrainingLoopConfig,
    make_tiny_cases,
    multi_positive_priority_loss,
    oracle_rollout,
    train_oracle_batches,
)


def _training_fixture(case_count: int = 8):
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    cases = make_tiny_cases(
        case_count=case_count,
        seed=122,
        generator_config=GeneratorConfig(min_nodes=8, max_nodes=9),
    )
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=14)
    batches = tuple(
        pack_rendered_cases(
            (case,),
            (renderer.render(case, row_permutation_seed=index),),
            schema=schema,
        )
        for index, case in enumerate(cases)
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
            dropout=0.0,
        )
    )
    return model, batches


def test_multi_positive_priority_does_not_choose_one_canonical_target() -> None:
    frontier = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)
    acceptable = torch.tensor([True, False, True, False, False])
    good = torch.tensor([3.0, -2.0, 2.0, 1.0, 1.0], requires_grad=True)
    bad = torch.tensor([-2.0, 3.0, -2.0, 1.0, 1.0], requires_grad=True)

    good_loss, good_count = multi_positive_priority_loss(
        good,
        acceptable,
        frontier,
        frontier_count=2,
    )
    bad_loss, bad_count = multi_positive_priority_loss(
        bad,
        acceptable,
        frontier,
        frontier_count=2,
    )

    assert good_count == bad_count == 1
    assert good_loss < bad_loss
    good_loss.backward()
    assert good.grad is not None
    assert good.grad[0] < 0
    assert good.grad[2] < 0


def test_oracle_rollout_is_finite_and_backpropagates() -> None:
    torch.manual_seed(7)
    model, batches = _training_fixture()
    result = oracle_rollout(
        model,
        batches[1],
        loss_config=SpiderLossConfig(),
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert result.metrics.candidate_count > 0
    assert result.metrics.termination_count == result.rounds
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_tiny_training_decreases_loss_and_saves_checkpoint(tmp_path) -> None:
    torch.manual_seed(19)
    model, batches = _training_fixture()
    checkpoint = tmp_path / "tiny.pt"
    result = train_oracle_batches(
        model,
        batches,
        loop_config=TrainingLoopConfig(
            steps=80,
            batch_size=4,
            learning_rate=0.003,
            seed=5,
            log_every=20,
        ),
        checkpoint_path=checkpoint,
    )

    assert result.records[-1].loss < result.records[0].loss
    assert result.final_metrics.expand_accuracy >= 0.80
    assert result.final_metrics.evidence_accuracy >= 0.80
    assert result.final_metrics.termination_accuracy >= 0.75
    assert checkpoint.exists()
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["final_metrics"]["candidate_expand_accuracy"] >= 0.80
