from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/run_spider_v0_3_termination_matrix.py")
    spec = importlib.util.spec_from_file_location(
        "spider_v03_termination_matrix",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(
    *,
    continuation: float,
    premature: float,
    autonomous: float,
    retention: float,
    unknown: float,
    risk: float,
    passed: bool = True,
):
    return {
        "pass": passed,
        "state_evaluation": {
            "continuation_recall": continuation,
            "premature_stop_rate": premature,
            "unknown_macro_recall": unknown,
        },
        "autonomous_retention": retention,
        "autonomous_evaluation": {
            "primary_autonomous_success": autonomous,
            "fixed_horizon_structural_success": 0.5,
            "rollout": {
                "risk_among_answered": risk,
                "false_answer_rate": 0.01,
            },
            "evidence": {"recall": 0.8},
            "evidence_pipeline": {
                "overall": {"exact_evidence_set_accuracy": 0.6}
            },
        },
    }


def test_matrix_selects_arm_passing_two_seeds_and_absolute_gates() -> None:
    module = _module()
    runs = {}
    for seed in module.SEEDS:
        runs[("T0", seed)] = _run(
            continuation=0.50,
            premature=0.50,
            autonomous=0.10,
            retention=0.20,
            unknown=0.50,
            risk=0.10,
        )
        runs[("T1", seed)] = _run(
            continuation=0.97 if seed != 1903 else 0.90,
            premature=0.03 if seed != 1903 else 0.10,
            autonomous=0.42,
            retention=0.90,
            unknown=0.75,
            risk=0.11,
        )
        runs[("T2", seed)] = _run(
            continuation=0.96,
            premature=0.04,
            autonomous=0.40,
            retention=0.88,
            unknown=0.65,
            risk=0.11,
        )

    decision = module._arm_decision(runs)

    assert decision["arms"]["T1"]["seed_passes"] == 2
    assert decision["arms"]["T1"]["eligible"]
    assert not decision["arms"]["T2"]["eligible"]
    assert decision["selected_arm"] == "T1"
    assert decision["termination_gate_passed"]


def test_answered_risk_degradation_blocks_otherwise_good_arm() -> None:
    module = _module()
    runs = {}
    for seed in module.SEEDS:
        runs[("T0", seed)] = _run(
            continuation=0.5,
            premature=0.5,
            autonomous=0.1,
            retention=0.2,
            unknown=0.5,
            risk=0.05,
        )
        for arm in ("T1", "T2"):
            runs[(arm, seed)] = _run(
                continuation=0.99,
                premature=0.01,
                autonomous=0.5,
                retention=0.95,
                unknown=0.8,
                risk=0.08,
            )

    decision = module._arm_decision(runs)

    assert decision["selected_arm"] is None
    assert not decision["termination_gate_passed"]
