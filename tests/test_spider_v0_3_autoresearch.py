from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/run_spider_v0_3_autoresearch.py")
    spec = importlib.util.spec_from_file_location(
        "spider_v03_autoresearch",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(
    *,
    recall: float,
    exact: float,
    precision: float = 0.8,
    coverage: float = 0.96,
    passed: bool = True,
):
    return {
        "pass": passed,
        "evidence_gate_metrics": {
            "recall": recall,
            "exact_set_accuracy": exact,
            "precision": precision,
            "scored_positive_coverage": coverage,
            "conditional_selection_recall": 0.9,
            "false_positives_per_case": 0.1,
        },
    }


def test_screen_advances_only_an_arm_winning_two_matched_seeds() -> None:
    module = _module()
    screen = {}
    for seed in module.SEEDS:
        screen[("E0", seed)] = _metrics(recall=0.60, exact=0.40)
    screen[("E1", 1701)] = _metrics(recall=0.66, exact=0.41)
    screen[("E1", 1802)] = _metrics(recall=0.67, exact=0.41)
    screen[("E1", 1903)] = _metrics(recall=0.61, exact=0.40)
    screen[("E2", 1701)] = _metrics(recall=0.68, exact=0.44)
    screen[("E2", 1802)] = _metrics(
        recall=0.70,
        exact=0.45,
        precision=0.70,
    )
    screen[("E2", 1903)] = _metrics(recall=0.61, exact=0.40)

    decision = module._screen_decision(screen)

    assert decision["arms"]["E1"]["eligible"]
    assert not decision["arms"]["E2"]["eligible"]
    assert decision["experimental_winner"] == "E1"
    assert decision["full_arms"] == ["E0", "E1"]


def test_guard_failed_run_cannot_count_as_seed_win() -> None:
    module = _module()
    screen = {}
    for seed in module.SEEDS:
        screen[("E0", seed)] = _metrics(recall=0.60, exact=0.40)
        screen[("E1", seed)] = _metrics(
            recall=0.70,
            exact=0.50,
            passed=seed == 1701,
        )
        screen[("E2", seed)] = _metrics(recall=0.60, exact=0.40)

    decision = module._screen_decision(screen)

    assert decision["arms"]["E1"]["seed_wins"] == 1
    assert decision["experimental_winner"] is None
