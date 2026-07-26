from __future__ import annotations

from pathlib import Path

from ..programs.splits import SplitSpec


DATASET_VERSION_V0_2 = "spider-programs-v0.2"


def validate_v0_1_split_access(
    spec: SplitSpec,
    *,
    allow_sealed: bool = False,
) -> None:
    if spec.generator_version != DATASET_VERSION_V0_2:
        raise ValueError(
            "Spider v0.1 commands accept only spider-programs-v0.2 splits"
        )
    if spec.sealed and not allow_sealed:
        raise PermissionError(
            "v0.2 sealed split is unavailable during search/calibration"
        )


def validate_v0_1_artifact_input(
    path: str | Path,
    *,
    allow_historical_checkpoint_diagnostic: bool = False,
) -> None:
    """Protect immutable v0 evidence from v0.1 selection commands."""

    resolved = Path(path).resolve()
    parts = resolved.parts
    historical = any(
        parts[index : index + 2] == ("artifacts", "spider_v0")
        for index in range(max(0, len(parts) - 1))
    )
    if historical and not allow_historical_checkpoint_diagnostic:
        raise PermissionError(
            "Spider v0 artifacts are immutable and unavailable to v0.1 search"
        )
