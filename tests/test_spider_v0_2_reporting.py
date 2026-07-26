import importlib.util
from pathlib import Path
import sys


def _module():
    path = Path("scripts/render_spider_v0_2_training_plots.py")
    spec = importlib.util.spec_from_file_location(
        "spider_v02_training_plots",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_learning_curve_svg_is_deterministic_and_labels_runs() -> None:
    module = _module()
    series = (
        module.HistorySeries(
            experiment_id="REC-recurrent-s1701-6k",
            model="recurrent",
            seed=1701,
            steps=(0, 100, 200),
            losses=(4.0, 2.0, 1.0),
        ),
        module.HistorySeries(
            experiment_id="REC-pooled-s1701-6k",
            model="pooled",
            seed=1701,
            steps=(0, 100, 200),
            losses=(3.0, 2.5, 2.0),
        ),
    )

    first = module.render_learning_curves_svg(series)
    second = module.render_learning_curves_svg(series)

    assert first == second
    assert first.startswith("<svg")
    assert "Training loss" in first
    assert "REC-recurrent-s1701-6k" in first
    assert "REC-pooled-s1701-6k" in first
    assert "polyline" in first


def test_structural_comparison_svg_uses_paired_and_ablation_results() -> None:
    module = _module()
    summary = {
        "paired": {
            "by_seed": {
                "1701": {
                    "recurrent_structural": 0.40,
                    "pooled_structural": 0.30,
                    "structural_delta": 0.10,
                },
                "1802": {
                    "recurrent_structural": 0.50,
                    "pooled_structural": 0.45,
                    "structural_delta": 0.05,
                },
                "1903": {
                    "recurrent_structural": 0.35,
                    "pooled_structural": 0.40,
                    "structural_delta": -0.05,
                },
            }
        },
        "state_use": {
            "by_intervention": {
                "reset": {"mean_degradation": 0.25},
                "detach": {"mean_degradation": 0.00},
                "shuffle": {"mean_degradation": 0.20},
                "pooled_current_node": {"mean_degradation": 0.15},
            }
        },
    }

    rendered = module.render_structural_comparison_svg(summary)

    assert rendered.startswith("<svg")
    assert "Fixed-horizon structural success" in rendered
    assert "Direct state-use degradation" in rendered
    assert "1701" in rendered
    assert "pooled current node" in rendered


def test_history_series_rejects_mismatched_lengths() -> None:
    module = _module()
    try:
        module.HistorySeries(
            experiment_id="bad",
            model="recurrent",
            seed=1,
            steps=(0, 1),
            losses=(1.0,),
        )
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("mismatched history lengths were accepted")
