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
    SparseWavefrontController,
    SpiderModel,
    SpiderModelConfig,
    calibrate_on_development_batches,
    controller_rollout,
    evaluate_closed_loop_batches,
    make_tiny_cases,
)


class _RecordingSpider(SpiderModel):
    def __init__(self, config: SpiderModelConfig) -> None:
        super().__init__(config)
        self.termination_hypothesis_counts: list[int] = []

    def termination_output(
        self,
        batch,
        hypotheses,
        evidence,
        controller_features=None,
    ):
        self.termination_hypothesis_counts.append(hypotheses.count)
        return super().termination_output(
            batch,
            hypotheses,
            evidence,
            controller_features,
        )


def _fixture():
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    cases = make_tiny_cases(
        case_count=8,
        seed=991,
        generator_config=GeneratorConfig(
            min_nodes=8,
            max_nodes=10,
            generator_version="spider-programs-v0.2",
        ),
    )
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=72)
    batches = tuple(
        pack_rendered_cases(
            (case,),
            (renderer.render(case, row_permutation_seed=index),),
            schema=schema,
        )
        for index, case in enumerate(cases)
    )
    permuted = tuple(
        pack_rendered_cases(
            (case,),
            (
                renderer.render(
                    case,
                    row_permutation_seed=10_000 + index,
                ),
            ),
            schema=schema,
        )
        for index, case in enumerate(cases)
    )
    model_config = SpiderModelConfig(
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
    controller_config = SparseControllerConfig(
        max_rounds=5,
        frontier_width=8,
        hypotheses_per_node=2,
        context_read_budget=4,
        evidence_selection_budget=4,
        search_budget=128,
        max_depth=8,
    )
    return batches, permuted, model_config, controller_config


def test_closed_loop_evaluation_reports_autonomous_primary_and_curves() -> None:
    torch.manual_seed(88)
    batches, permuted, model_config, controller_config = _fixture()
    model = SpiderModel(model_config).eval()
    calibration = calibrate_on_development_batches(
        model,
        batches,
        controller_config=controller_config,
    )
    report = evaluate_closed_loop_batches(
        model,
        batches,
        split="validation_id",
        controller_config=controller_config,
        evidence_threshold=calibration.threshold,
        permuted_batches=permuted,
    )

    assert 0.0 <= report.primary_autonomous_success <= 1.0
    assert report.evidence["positive_label_count"] >= 0
    assert report.evidence["negative_label_count"] > 0
    assert report.evidence["precision_recall_curve"]
    assert len(report.termination["overall_confusion"]) == 6
    assert set(report.per_family) == {
        "lookup",
        "reachability",
        "latest_valid",
        "corroboration",
    }
    assert report.invariance["deterministic_replay_mismatches"] == 0
    assert report.invariance["row_permutation_decision_mismatches"] == 0


def test_termination_observes_post_transition_state_in_all_modes() -> None:
    torch.manual_seed(89)
    batches, _, model_config, controller_config = _fixture()
    model = _RecordingSpider(model_config).eval()
    rollout = controller_rollout(
        model,
        batches[0],
        controller_config=controller_config,
        action_schedule=ActionSchedule.oracle_only(),
        randomizer=random.Random(0),
    )
    training_counts = tuple(model.termination_hypothesis_counts)
    assert training_counts == tuple(
        diagnostic.termination_hypothesis_count
        for diagnostic in rollout.diagnostics
    )

    model.termination_hypothesis_counts.clear()
    result = SparseWavefrontController(controller_config).run(
        model,
        batches[0],
    )
    runtime_counts = tuple(model.termination_hypothesis_counts)
    assert runtime_counts == tuple(
        len(diagnostic.frontier_candidate_indices)
        for diagnostic in result.action_diagnostics
    )
