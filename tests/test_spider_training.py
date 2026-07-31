from __future__ import annotations

import random
import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    GeneratorConfig,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ActionSchedule,
    SparseControllerConfig,
    SpiderLossConfig,
    SpiderModel,
    SpiderModelConfig,
    TrainingLoopConfig,
    CandidateOutputs,
    behavioural_consistency_loss,
    evaluate_batches,
    make_tiny_cases,
    mixed_rollout,
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


def test_behavioural_consistency_aligns_actions_not_hidden_coordinates() -> None:
    def outputs(offset: float) -> CandidateOutputs:
        logits = (
            torch.tensor([0.2, -0.4]) + offset
        ).detach().requires_grad_()
        state = (
            torch.randn(2, 3, 4) + 10.0 * offset
        ).detach().requires_grad_()
        return CandidateOutputs(
            next_path_state=state,
            priority_logits=logits,
            expand_logits=logits,
            context_logits=logits,
            evidence_logits=logits,
            remaining_cost=logits.square(),
            support_logits=logits,
            conflict_logits=logits,
        )

    first = outputs(0.0)
    equivalent = outputs(0.0)
    shifted = outputs(1.0)
    indices = torch.tensor([0, 1])
    same = behavioural_consistency_loss(
        first,
        equivalent,
        indices,
        indices,
    )
    different = behavioural_consistency_loss(
        first,
        shifted,
        indices,
        indices,
    )

    assert same.raw < 1e-7
    assert different.raw > same.raw
    different.weighted.backward()
    assert first.priority_logits.grad is not None


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


def test_mixed_model_rollout_produces_hard_negative_training_signal() -> None:
    torch.manual_seed(13)
    model, batches = _training_fixture()
    result = mixed_rollout(
        model,
        batches[1],
        oracle_fraction=0.0,
        randomizer=random.Random(4),
        max_rounds=4,
    )
    result.loss.backward()

    assert result.rounds >= 1
    assert torch.isfinite(result.loss)
    assert result.metrics.candidate_count > 0
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
        checkpoint_every=40,
    )

    assert result.records[-1].loss < result.records[0].loss
    assert result.final_metrics.expand_accuracy >= 0.80
    assert result.final_metrics.evidence_accuracy >= 0.80
    assert result.final_metrics.termination_accuracy >= 0.75
    assert checkpoint.exists()
    assert (tmp_path / "tiny_step_000040.pt").exists()
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["final_metrics"]["candidate_expand_accuracy"] >= 0.80


def test_evaluation_reports_rollout_costs_and_invariance() -> None:
    torch.manual_seed(23)
    model, batches = _training_fixture()
    report = evaluate_batches(
        model.eval(),
        batches[:2],
        split="test_fixture",
        controller_config=SparseControllerConfig(
            max_rounds=2,
            frontier_width=8,
            hypotheses_per_node=2,
            context_read_budget=2,
            search_budget=64,
            max_depth=4,
        ),
        permuted_batches=batches[:2],
        invariance_sample_limit=2,
    )

    assert report.case_count == 2
    assert report.teacher_forced["candidate_mrr"] >= 0.0
    assert report.rollout["trace_validity"] == 1.0
    assert report.efficiency["mean_arcs_scored"] > 0
    assert report.invariance["deterministic_replay_mismatches"] == 0
    assert report.invariance["row_permutation_decision_mismatches"] == 0


def test_paused_training_resumes_to_identical_model_state(tmp_path) -> None:
    torch.manual_seed(29)
    uninterrupted, batches = _training_fixture()
    initial_state = {
        name: value.detach().clone()
        for name, value in uninterrupted.state_dict().items()
    }
    initial_rng = torch.get_rng_state()
    loop = TrainingLoopConfig(
        steps=6,
        batch_size=2,
        learning_rate=0.001,
        seed=17,
        log_every=3,
        action_schedule=(
            ActionSchedule.oracle_only(),
            ActionSchedule.model_only(),
        ),
    )

    torch.set_rng_state(initial_rng)
    full = train_oracle_batches(
        uninterrupted,
        batches,
        loop_config=loop,
    )

    paused = SpiderModel(uninterrupted.config)
    paused.load_state_dict(initial_state)
    checkpoint = tmp_path / "resumable.pt"
    torch.set_rng_state(initial_rng)
    partial = train_oracle_batches(
        paused,
        batches,
        loop_config=loop,
        checkpoint_path=checkpoint,
        stop_after_steps=3,
    )
    resumed = SpiderModel(uninterrupted.config)
    resumed_result = train_oracle_batches(
        resumed,
        batches,
        loop_config=loop,
        checkpoint_path=checkpoint,
        resume_checkpoint=checkpoint,
    )

    assert full.completed_steps == 6
    assert partial.completed_steps == 3
    assert resumed_result.completed_steps == 6
    assert resumed_result.resumed_from_step == 3
    for name, expected in uninterrupted.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[name]), name
    assert (
        full.action_source_counts
        == resumed_result.action_source_counts
    )
    assert full.training_examples == resumed_result.training_examples
