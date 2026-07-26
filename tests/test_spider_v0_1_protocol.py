from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hippocampus.programs import (
    build_split_manifest,
    default_split_specs,
    default_split_specs_v0_2,
)
from hippocampus.spider import (
    authorize_v0_2_sealed_evaluation,
    validate_v0_1_artifact_input,
    validate_v0_1_split_access,
)


def test_v0_1_search_rejects_old_and_sealed_splits() -> None:
    with pytest.raises(ValueError, match="v0.2"):
        validate_v0_1_split_access(default_split_specs()[0])
    sealed = next(
        spec for spec in default_split_specs_v0_2() if spec.sealed
    )
    with pytest.raises(PermissionError, match="sealed"):
        validate_v0_1_split_access(sealed)
    validate_v0_1_split_access(sealed, allow_sealed=True)


def test_v0_1_search_rejects_historical_artifact_inputs() -> None:
    old_checkpoint = Path(
        "artifacts/spider_v0/autoresearch/runs/"
        "E003-recurrent-standard/checkpoint.pt"
    )
    with pytest.raises(PermissionError, match="immutable"):
        validate_v0_1_artifact_input(old_checkpoint)
    validate_v0_1_artifact_input(
        old_checkpoint,
        allow_historical_checkpoint_diagnostic=True,
    )
    validate_v0_1_artifact_input(
        "artifacts/spider_v0_1/runs/E004/checkpoint.pt"
    )


def test_sealed_evaluation_is_hash_bound_explicit_and_one_shot(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts" / "spider_v0_1"
    artifact_root.mkdir(parents=True)
    config = artifact_root / "finalist.json"
    checkpoint = artifact_root / "checkpoint.pt"
    config.write_text(
        json.dumps(
            {
                "name": "finalist",
                "dataset": {"version": "spider-programs-v0.2"},
            }
        )
    )
    checkpoint.write_bytes(b"frozen checkpoint")

    sealed = next(
        spec
        for spec in default_split_specs_v0_2()
        if spec.name == "test_sealed_v0_2"
    )
    sealed_hash = build_split_manifest(sealed).sha256
    split_index = tmp_path / "MANIFEST_INDEX.json"
    split_index.write_text(
        json.dumps(
            {
                "aggregate_sha256": "aggregate",
                "generator_version": "spider-programs-v0.2",
                "sealed_cases_materialised": False,
                "split_hashes": {"test_sealed_v0_2": sealed_hash},
            }
        )
    )

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    finalist = tmp_path / "FINALIST_MANIFEST.json"
    finalist.write_text(
        json.dumps(
            {
                "experiment_id": "finalist",
                "config_path": str(config.relative_to(tmp_path)),
                "config_sha256": sha(config),
                "checkpoint_path": str(checkpoint.relative_to(tmp_path)),
                "checkpoint_sha256": sha(checkpoint),
                "dataset_version": "spider-programs-v0.2",
                "dataset_split_digest": "aggregate",
                "sealed_split_sha256": sealed_hash,
                "evidence_threshold": 0.42,
                "sealed_opened": False,
                "sealed_evaluation_protocol": {
                    "calibration_split": "validation_id",
                    "calibration_during_evaluation": False,
                    "renderer_seed": 91337,
                    "row_seed_offset": 300000,
                    "permuted_row_seed_offset": 400000,
                    "invariance_sample_limit": 32,
                },
            }
        )
    )
    marker = tmp_path / "SEALED_ACCESS.json"
    output = tmp_path / "SEALED_EVALUATION.json"

    with pytest.raises(PermissionError, match="explicit"):
        authorize_v0_2_sealed_evaluation(
            finalist,
            split_index,
            access_marker_path=marker,
            output_path=output,
            repository_root=tmp_path,
        )
    authorization = authorize_v0_2_sealed_evaluation(
        finalist,
        split_index,
        access_marker_path=marker,
        output_path=output,
        allow_sealed=True,
        repository_root=tmp_path,
    )
    assert authorization.evidence_threshold == pytest.approx(0.42)
    assert authorization.sealed_split_sha256 == sealed_hash

    marker.write_text("{}")
    with pytest.raises(FileExistsError, match="already"):
        authorize_v0_2_sealed_evaluation(
            finalist,
            split_index,
            access_marker_path=marker,
            output_path=output,
            allow_sealed=True,
            repository_root=tmp_path,
        )
