from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/aggregate_spider_v0_2_diagnostics.py")
    spec = importlib.util.spec_from_file_location("spider_v02_aggregate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_rule_requires_material_effect_in_two_seeds() -> None:
    module = _module()
    failed = module._state_rule(
        {
            "reset": {1701: 0.06, 1802: 0.01, 1903: 0.04},
            "shuffle": {1701: 0.02, 1802: 0.02, 1903: 0.02},
        }
    )
    passed = module._state_rule(
        {
            "reset": {1701: 0.08, 1802: 0.07, 1903: 0.01},
            "shuffle": {1701: 0.02, 1802: 0.02, 1903: 0.02},
        }
    )

    assert not failed["material_state_use"]
    assert passed["material_state_use"]


def test_weighted_state_score_uses_only_pre_registered_standard_splits() -> None:
    module = _module()
    run = {
        "state_ablation_reports": {
            "none": {
                "validation_id": {
                    "case_count": 3,
                    "structural_success": 1.0,
                },
                "validation_path_length_ood": {
                    "case_count": 1,
                    "structural_success": 0.0,
                },
                "validation_recurrence_necessity": {
                    "case_count": 100,
                    "structural_success": 0.0,
                },
            }
        }
    }

    assert module._weighted_state_score(run, "none") == pytest.approx(0.75)
