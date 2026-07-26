from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/aggregate_spider_v0_2_training.py")
    spec = importlib.util.spec_from_file_location(
        "spider_v02_training_aggregate",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verifier_module():
    path = Path("scripts/verify_spider_v0_2_recurrence_run.py")
    spec = importlib.util.spec_from_file_location(
        "spider_v02_training_verify",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(
    model: str,
    seed: int,
    structural_success: float,
    *,
    reset: float | None = None,
    shuffle: float | None = None,
) -> dict[str, object]:
    state_ablations: dict[str, dict[str, object]] = {}
    if model == "recurrent":
        intact = structural_success
        state_ablations = {
            "none": {
                "structural_success": intact,
                "evidence_exact_set_accuracy": intact,
                "evidence_recall": intact,
                "valid_path_rate": 1.0,
                "mean_rounds": 6.0,
                "replay_mismatches": 0,
                "row_permutation_mismatches": 0,
            },
            "reset": {
                "structural_success": reset if reset is not None else intact,
                "evidence_exact_set_accuracy": 0.0,
                "evidence_recall": 0.0,
                "valid_path_rate": 1.0,
                "mean_rounds": 6.0,
                "replay_mismatches": 0,
                "row_permutation_mismatches": 0,
            },
            "detach": {
                "structural_success": intact,
                "evidence_exact_set_accuracy": intact,
                "evidence_recall": intact,
                "valid_path_rate": 1.0,
                "mean_rounds": 6.0,
                "replay_mismatches": 0,
                "row_permutation_mismatches": 0,
            },
            "shuffle": {
                "structural_success": (
                    shuffle if shuffle is not None else intact
                ),
                "evidence_exact_set_accuracy": 0.0,
                "evidence_recall": 0.0,
                "valid_path_rate": 1.0,
                "mean_rounds": 6.0,
                "replay_mismatches": 0,
                "row_permutation_mismatches": 0,
            },
            "pooled_current_node": {
                "structural_success": intact,
                "evidence_exact_set_accuracy": intact,
                "evidence_recall": intact,
                "valid_path_rate": 1.0,
                "mean_rounds": 6.0,
                "replay_mismatches": 0,
                "row_permutation_mismatches": 0,
            },
        }
    return {
        "experiment_id": f"REC-{model}-s{seed}-6k",
        "model": model,
        "seed": seed,
        "steps": 6000,
        "status": "accepted",
        "failure_reason": None,
        "sealed_access_count": 0,
        "source_commit": (
            "acb533666d481daf9b6fb56562d69a5dd78c5e0e"
        ),
        "runtime_seconds": 10.0,
        "training_runtime_seconds": 8.0,
        "parameter_count": 100,
        "primary_structural_success": structural_success,
        "primary_final_autonomous_success": 0.0,
        "peak_cuda_memory_bytes": 1,
        "checkpoints": {},
        "published_checkpoints": {},
        "reports": {
            "primary": {
                "case_count": 128,
                "evidence": {
                    "exact_set_accuracy": structural_success,
                    "precision": structural_success,
                    "recall": structural_success,
                    "f1": structural_success,
                },
                "rollout": {
                    "exact_valid_path_rate": 1.0,
                    "trace_validity": 1.0,
                    "termination_accuracy": 0.0,
                    "false_answer_rate": 0.0,
                    "semantic_invalid_expansion_rate": 0.0,
                },
                "efficiency": {
                    "mean_rounds": 6.0,
                    "mean_arcs_scored": 20.0,
                    "mean_contexts_read": 0.0,
                },
                "invariance": {
                    "deterministic_replay_mismatches": 0,
                    "row_permutation_decision_mismatches": 0,
                    "maximum_score_delta": 0.0,
                },
            },
            "state_ablations": state_ablations,
        },
    }


def test_summary_applies_frozen_paired_and_state_use_rules() -> None:
    module = _module()
    records = []
    for seed, recurrent, pooled in (
        (1701, 0.50, 0.45),
        (1802, 0.55, 0.50),
        (1903, 0.48, 0.49),
    ):
        records.append(
            _record(
                "recurrent",
                seed,
                recurrent,
                reset=recurrent - 0.08,
                shuffle=recurrent - 0.02,
            )
        )
        records.append(_record("pooled", seed, pooled))

    summary = module.build_summary(records, completed_at="frozen")

    assert summary["decision"]["recurrent_advantage"]
    assert summary["decision"]["material_state_use"]
    assert summary["paired"]["recurrent_seed_wins"] == 2
    assert summary["paired"]["mean_structural_delta"] == pytest.approx(0.03)
    assert summary["state_use"]["by_intervention"]["reset"][
        "seeds_at_or_above_0_05"
    ] == 3


def test_state_use_rule_ignores_detach_forward_equivalence() -> None:
    module = _module()
    degradations = {
        "reset": {1701: 0.01, 1802: 0.01, 1903: 0.01},
        "detach": {1701: 1.0, 1802: 1.0, 1903: 1.0},
        "shuffle": {1701: 0.02, 1802: 0.02, 1903: 0.02},
        "pooled_current_node": {1701: 0.0, 1802: 0.0, 1903: 0.0},
    }

    assert not module.state_use_decision(degradations)["material_state_use"]


def test_record_validation_fails_closed_on_sealed_access() -> None:
    module = _module()
    record = _record("pooled", 1701, 0.5)
    record["sealed_access_count"] = 1

    with pytest.raises(RuntimeError, match="sealed"):
        module.validate_record(record)


def test_summary_requires_the_exact_six_run_matrix() -> None:
    module = _module()
    records = [
        _record("recurrent", seed, 0.5)
        for seed in module.SEEDS
    ]

    with pytest.raises(RuntimeError, match="frozen run set"):
        module.build_summary(records, completed_at="frozen")


def test_artifact_manifest_verification_detects_byte_tampering(
    tmp_path: Path,
) -> None:
    verifier = _verifier_module()
    run = tmp_path / "REC-recurrent-s1701-6k"
    run.mkdir()
    payload = run / "payload.bin"
    payload.write_bytes(b"original")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {
        "experiment_id": run.name,
        "file_count_excluding_manifest": 1,
        "files": {
            "payload.bin": {
                "bytes": payload.stat().st_size,
                "sha256": digest,
            }
        },
    }
    (run / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest)
    )

    assert verifier.verify_artifact_manifest(run) == manifest
    payload.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256"):
        verifier.verify_artifact_manifest(run)


def test_drive_entry_validation_rejects_duplicate_file_ids() -> None:
    module = _module()
    entry = {
        "bytes": 12,
        "sha256": "f" * 64,
        "drive_parent_verified": True,
        "drive_parent_id": "folder",
        "drive_size_verified": True,
        "drive_id": "drive-file-1",
        "drive_url": "https://drive.google.com/file/d/drive-file-1/view",
    }
    drive_ids: set[str] = set()

    module._validate_drive_entry(
        entry,
        expected_bytes=12,
        expected_sha256="f" * 64,
        folder_id="folder",
        drive_ids=drive_ids,
        field="first",
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        module._validate_drive_entry(
            entry,
            expected_bytes=12,
            expected_sha256="f" * 64,
            folder_id="folder",
            drive_ids=drive_ids,
            field="second",
        )
