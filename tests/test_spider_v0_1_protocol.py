from __future__ import annotations

from pathlib import Path

import pytest

from hippocampus.programs import default_split_specs, default_split_specs_v0_2
from hippocampus.spider import (
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
