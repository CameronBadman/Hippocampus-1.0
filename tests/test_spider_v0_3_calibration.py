from __future__ import annotations

import pytest
import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    GeneratorConfig,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ControllerExecutionPolicy,
    SparseControllerConfig,
    SpiderModel,
    SpiderModelConfig,
    calibrate_closed_loop_evidence,
    fit_temperature_scaling,
    make_tiny_cases,
)


def _fixture():
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    cases = make_tiny_cases(
        case_count=8,
        seed=12_300,
        generator_config=GeneratorConfig(
            min_nodes=8,
            max_nodes=9,
            generator_version="spider-programs-v0.2",
        ),
    )
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=91)
    batches = tuple(
        pack_rendered_cases(
            (case,),
            (renderer.render(case, row_permutation_seed=index),),
            schema=schema,
        )
        for index, case in enumerate(cases[:2])
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
    ).eval()
    controller = SparseControllerConfig(
        max_rounds=4,
        frontier_width=8,
        hypotheses_per_node=2,
        context_read_budget=4,
        evidence_selection_budget=4,
        search_budget=128,
        max_depth=6,
    )
    return model, batches, controller


def test_temperature_scaling_is_deterministic_and_gate_controlled() -> None:
    logits = torch.tensor([8.0, 6.0, -8.0, -6.0])
    labels = torch.tensor([True, False, False, True])

    first = fit_temperature_scaling(logits, labels)
    repeated = fit_temperature_scaling(logits, labels)

    assert first == repeated
    assert 0.25 <= first.fitted_temperature <= 4.0
    assert first.applied_temperature in {
        1.0,
        first.fitted_temperature,
    }
    if first.accepted:
        assert first.fitted_nll < first.baseline_nll
        assert first.fitted_brier <= first.baseline_brier


def test_closed_loop_calibration_uses_exact_set_curve_on_dev_only() -> None:
    torch.manual_seed(101)
    model, batches, controller = _fixture()
    calibration = calibrate_closed_loop_evidence(
        model,
        batches,
        controller_config=controller,
        split_name="development_calibration",
        dataset_version="spider-programs-v0.2",
        execution_policy=ControllerExecutionPolicy.fixed(2),
        coarse_thresholds=(0.25, 0.75),
        fine_radius=0.0,
    )

    assert calibration.source_case_count == 2
    assert len(calibration.source_case_hash) == 64
    assert len(calibration.curve) == 2
    assert 0.0 < calibration.threshold < 1.0
    assert (
        calibration.selected
        == max(
            calibration.curve,
            key=lambda point: (
                point.exact_set_accuracy,
                point.recall,
                point.precision,
                -point.false_positives_per_case,
                point.calibrated_probability_threshold,
            ),
        )
    )

    with pytest.raises(ValueError, match="sealed"):
        calibrate_closed_loop_evidence(
            model,
            batches,
            controller_config=controller,
            split_name="test_sealed_v0_2",
            dataset_version="spider-programs-v0.2",
            coarse_thresholds=(0.5,),
            fine_radius=0.0,
        )
