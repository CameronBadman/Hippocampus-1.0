from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from hippocampus.spider import (
    EvidenceCandidateObservation,
    EvidencePipelineCaseReport,
    EvidenceRequirement,
    aggregate_evidence_pipeline,
    load_experiment,
)


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_spider_v0_4_set_decoding.py"
    spec = importlib.util.spec_from_file_location("spider_v04_phase_f", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(*, exact: float, precision: float, recall: float, error: float):
    return {
        "primary_metric": {
            "exact_evidence_set_accuracy": exact,
            "precision": precision,
            "recall": recall,
            "mean_absolute_cardinality_error": error,
        }
    }


def _report(case_id: str, *, false_positives: int) -> EvidencePipelineCaseReport:
    requirement = EvidenceRequirement(None, None, 1)
    candidates = [
        EvidenceCandidateObservation(
            round_index=0,
            arc_id=1,
            edge_id=1,
            source_node=0,
            destination_node=1,
            logit=2.0,
            pre_context_logit=2.0,
            required=True,
            selected=True,
            recorded=True,
            frontier_selected=False,
        )
    ]
    if false_positives:
        candidates.append(
            EvidenceCandidateObservation(
                round_index=0,
                arc_id=2,
                edge_id=2,
                source_node=0,
                destination_node=2,
                logit=1.0,
                pre_context_logit=1.0,
                required=False,
                selected=True,
                recorded=True,
                frontier_selected=False,
            )
        )
    return EvidencePipelineCaseReport(
        case_id=case_id,
        family="lookup",
        horizon=1,
        requirements=(requirement,),
        requirement_observations=(),
        candidate_observations=tuple(candidates),
        exact_set_accuracy=float(false_positives == 0),
        true_positives=1,
        false_positives=false_positives,
        false_negatives=0,
        predicted_cardinality=1 + false_positives,
        required_cardinality=1,
        average_precision=1.0,
        worst_positive_rank=1,
        minimum_positive_negative_margin=1.0,
    )


def test_phase_f_configs_change_only_registered_decoder_axes() -> None:
    configs = {
        arm: json.loads(
            (ROOT / f"configs/spider_v0_4/phase_f_{arm}.json").read_text()
        )
        for arm in ("F0", "F1", "F2", "F3")
    }
    for config in configs.values():
        assert config["dataset"]["protocol"] == "spider-v0.4-set-decoding"
        assert config["dataset"]["version"] == (
            "spider-programs-v0.4.1-aligned-evidence-dev"
        )
        assert config["training"]["steps"] == 2000
        assert config["training"]["seed"] == 1701

    assert configs["F0"]["controller"]["evidence_selection_policy"] == "threshold"
    assert configs["F1"]["controller"]["evidence_selection_policy"] == "learned_null"
    assert configs["F2"]["controller"]["evidence_selection_policy"] == "cardinality"
    assert configs["F3"]["controller"]["evidence_selection_policy"] == "null_cardinality"
    assert not configs["F0"]["model"]["use_evidence_null"]
    assert configs["F1"]["model"]["use_evidence_null"]
    assert configs["F2"]["model"]["use_evidence_cardinality"]
    assert configs["F3"]["model"]["use_evidence_null"]
    assert configs["F3"]["model"]["use_evidence_cardinality"]


def test_phase_f_control_resolves_identically_to_historical_d0() -> None:
    historical = load_experiment(ROOT / "configs/spider_v0_4/phase_d_D0.json")
    control = load_experiment(ROOT / "configs/spider_v0_4/phase_f_F0.json")

    assert historical.schema == control.schema
    assert historical.model_config == control.model_config
    assert historical.controller_config == control.controller_config
    assert historical.training_config == control.training_config
    assert historical.loss_config == control.loss_config
    assert historical.raw["renderer"] == control.raw["renderer"]


def test_phase_f_gate_requires_all_registered_improvements() -> None:
    module = _module()
    control = _metrics(exact=0.70, precision=0.92, recall=0.60, error=0.30)
    passing = _metrics(exact=0.75, precision=0.90, recall=0.63, error=0.29)

    assert module._seed_gate(control, passing)["advances"]
    for failing in (
        _metrics(exact=0.749, precision=0.90, recall=0.64, error=0.29),
        _metrics(exact=0.76, precision=0.90, recall=0.629, error=0.29),
        _metrics(exact=0.76, precision=0.899, recall=0.64, error=0.29),
        _metrics(exact=0.76, precision=0.90, recall=0.64, error=0.30),
    ):
        assert not module._seed_gate(control, failing)["advances"]


def test_phase_f_resume_distinguishes_training_selection_and_evaluation(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "run"
    output.mkdir()
    partial = output / "checkpoint_step_001500.pt"
    partial.touch()
    assert module._interrupted_stage(output) == ("training", partial)

    (output / "checkpoint.pt").touch()
    assert module._interrupted_stage(output) == ("selection", None)

    (output / "evaluation_pause.json").touch()
    assert module._interrupted_stage(output) == ("evaluation", None)


def test_phase_f_campaign_lock_rejects_concurrent_orchestrator(
    tmp_path: Path,
) -> None:
    module = _module()

    with module._campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another Phase F orchestrator"):
            with module._campaign_lock(tmp_path):
                pass


def test_pipeline_reports_mean_absolute_cardinality_error() -> None:
    exact = _report("exact", false_positives=0)
    over = _report("over", false_positives=1)

    overall = aggregate_evidence_pipeline((exact, over))["overall"]

    assert overall["mean_absolute_cardinality_error"] == pytest.approx(0.5)
