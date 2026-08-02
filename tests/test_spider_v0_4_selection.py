from __future__ import annotations

import pytest
import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    SyntheticManifoldRenderer,
    default_aligned_dev_specs,
    generate_aligned_dev_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ControllerExecutionPolicy,
    PooledScorer,
    SparseControllerConfig,
    SpiderModelConfig,
    fast_calibrate_closed_loop_evidence,
)


def _fixture():
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    cases = generate_aligned_dev_cases(
        default_aligned_dev_specs()[2],
        limit=4,
    )
    renderer = SyntheticManifoldRenderer(
        schema,
        query_dim=8,
        seed=41,
        geometry="orthogonal_aligned",
    )
    batches = tuple(
        pack_rendered_cases(
            (case,),
            (renderer.render(case, row_permutation_seed=index),),
            schema=schema,
        )
        for index, case in enumerate(cases)
    )
    model = PooledScorer(
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


def test_fast_calibration_shortlists_then_runs_exact_controller() -> None:
    torch.manual_seed(101)
    model, batches, controller = _fixture()
    result = fast_calibrate_closed_loop_evidence(
        model,
        batches,
        controller_config=controller,
        split_name="calibration",
        dataset_version="spider-programs-v0.4-aligned-dev",
        precision_floor=0.0,
        coverage_floor=0.0,
        execution_policy=ControllerExecutionPolicy.fixed(2),
        fit_temperature=False,
        approximate_thresholds=(0.25, 0.5, 0.75),
        exact_candidate_count=2,
    )

    assert result.calibration.source_case_count == 4
    assert len(result.approximate_curve) == 3
    assert len(result.calibration.curve) == 2
    assert len(result.exact_candidate_thresholds) == 2
    assert result.calibration.temperature.applied_temperature == 1.0
    assert not result.temperature_fitted
    assert result.calibration.selected in result.calibration.curve


def test_fast_calibration_rejects_sealed_names_before_execution() -> None:
    model, batches, controller = _fixture()

    with pytest.raises(ValueError, match="sealed"):
        fast_calibrate_closed_loop_evidence(
            model,
            batches,
            controller_config=controller,
            split_name="sealed_v0_4",
            dataset_version="spider-programs-v0.4-aligned-dev",
        )
