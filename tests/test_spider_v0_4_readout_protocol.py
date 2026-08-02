from __future__ import annotations

import json
import runpy
from pathlib import Path

from hippocampus.spider import load_experiment


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1701, 1802, 1903)
RUNNER = runpy.run_path(str(ROOT / "scripts/run_spider_v0_4_readout.py"))
_arm_gate = RUNNER["_arm_gate"]
_screen_decision = RUNNER["_screen_decision"]


def _metrics(
    *,
    exact: float,
    precision: float,
    recall: float,
    average_precision: float,
    step: int = 500,
) -> dict[str, object]:
    return {
        "selected_step": step,
        "primary_metric": {
            "exact_evidence_set_accuracy": exact,
            "precision": precision,
            "recall": recall,
            "scored_positive_coverage": 0.99,
            "macro_average_precision": average_precision,
            "false_positives_per_case": 0.1,
            "mean_worst_positive_rank": 1.5,
            "constraint_satisfied": precision >= 0.90,
        },
    }


def test_phase_d_configs_change_only_registered_model_readout_axes() -> None:
    expected = {
        "D0": ("pooled", "shared", 1),
        "D1": ("pooled", "dedicated_pooled", 1),
        "D2": ("spider", "shared", 2),
        "D3": ("spider", "dedicated_pooled", 2),
        "D4": ("spider", "slot_aware", 2),
    }
    reference_loss = None
    reference_controller = None
    for arm, model_axes in expected.items():
        path = ROOT / f"configs/spider_v0_4/phase_d_{arm}.json"
        raw = json.loads(path.read_text())
        experiment = load_experiment(path)

        assert raw["dataset"]["version"] == (
            "spider-programs-v0.4.1-aligned-evidence-dev"
        )
        assert raw["dataset"]["training_case_count"] == 512
        assert raw["renderer"]["geometry"] == "orthogonal_aligned"
        assert experiment.training_config.steps == 2000
        assert (
            raw["model"]["kind"],
            experiment.model_config.evidence_readout,
            experiment.model_config.num_blocks,
        ) == model_axes
        reference_loss = reference_loss or raw["loss"]
        reference_controller = reference_controller or raw["controller"]
        assert raw["loss"] == reference_loss
        assert raw["controller"] == reference_controller


def test_readout_gate_requires_two_matched_seed_wins() -> None:
    results = {}
    for seed in SEEDS:
        results[("D0", seed)] = _metrics(
            exact=0.50,
            precision=0.92,
            recall=0.40,
            average_precision=0.70,
        )
        results[("D1", seed)] = _metrics(
            exact=0.56 if seed != 1903 else 0.51,
            precision=0.91,
            recall=0.42,
            average_precision=0.71,
        )

    decision = _arm_gate(
        results,
        control_arm="D0",
        candidate_arm="D1",
    )

    assert decision["seed_wins"] == 2
    assert decision["advances"] is True


def test_screen_decision_retains_controls_when_ablation_fails() -> None:
    results = {}
    for seed in SEEDS:
        for arm in ("D0", "D2"):
            results[(arm, seed)] = _metrics(
                exact=0.60,
                precision=0.93,
                recall=0.50,
                average_precision=0.75,
            )
        for arm in ("D1", "D3", "D4"):
            results[(arm, seed)] = _metrics(
                exact=0.59,
                precision=0.93,
                recall=0.49,
                average_precision=0.74,
            )

    decision = _screen_decision(results)

    assert decision["pooled_finalist"] == "D0"
    assert decision["spider_finalist"] == "D2"
    assert decision["full_run_arms"] == ["D0", "D2"]
    assert decision["sealed_access_count"] == 0
