from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path("scripts/run_spider_v0_4_autoresearch.py")
    spec = importlib.util.spec_from_file_location("spider_v04_autoresearch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(
    *,
    lookup: float,
    reachability: float,
    exact: float,
    precision: float,
    passed: bool = True,
):
    return {
        "pass": passed,
        "primary_metric": {
            "exact_evidence_set_accuracy": exact,
            "precision": precision,
            "recall": 0.5,
            "scored_positive_coverage": 0.99,
            "macro_average_precision": 0.7,
            "false_positives_per_case": 0.1,
        },
        "per_family": {
            "lookup": {"recall": lookup},
            "reachability": {"recall": reachability},
        },
    }


def test_phase_b_advances_only_after_two_matched_renderer_wins() -> None:
    module = _module()
    results = {}
    for seed in module.SEEDS:
        results[("B0", seed)] = _metrics(
            lookup=0.10,
            reachability=0.20,
            exact=0.30,
            precision=0.92,
        )
        results[("B1", seed)] = _metrics(
            lookup=0.50,
            reachability=0.50,
            exact=0.50,
            precision=0.91,
        )
        results[("B2", seed)] = _metrics(
            lookup=0.41 if seed != 1903 else 0.39,
            reachability=0.41,
            exact=0.40,
            precision=0.905,
        )

    decision = module._renderer_decision(results)

    assert decision["seed_wins"] == 2
    assert decision["B2_advances"]
    assert decision["next_phase"] == "C"


def test_phase_b_precision_regression_blocks_a_seed_win() -> None:
    module = _module()
    results = {}
    for seed in module.SEEDS:
        results[("B0", seed)] = _metrics(
            lookup=0.0,
            reachability=0.0,
            exact=0.0,
            precision=0.95,
        )
        results[("B1", seed)] = results[("B0", seed)]
        results[("B2", seed)] = _metrics(
            lookup=1.0,
            reachability=1.0,
            exact=1.0,
            precision=0.92,
        )

    decision = module._renderer_decision(results)

    assert decision["seed_wins"] == 0
    assert not decision["B2_advances"]


def test_only_b1_and_b2_create_new_renderer_training_runs() -> None:
    b0 = json.loads(Path("configs/spider_v0_4/phase_b_B0.json").read_text())
    b1 = json.loads(Path("configs/spider_v0_4/phase_b_B1.json").read_text())
    b2 = json.loads(Path("configs/spider_v0_4/phase_b_B2.json").read_text())

    assert "historical_checkpoint_template" in b0
    assert "historical_checkpoint_template" not in b1
    assert "historical_checkpoint_template" not in b2
    assert {b1["renderer"]["geometry"], b2["renderer"]["geometry"]} == {
        "shared_additive",
        "orthogonal_aligned",
    }


def test_single_experiment_filters_are_available_for_bounded_runs(
    monkeypatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_spider_v0_4_autoresearch.py",
            "--arm",
            "B2",
            "--seed",
            "1802",
        ],
    )

    args = module.parse_args()

    assert args.arm == "B2"
    assert args.seed == 1802
