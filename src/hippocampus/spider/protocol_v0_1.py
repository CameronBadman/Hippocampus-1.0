from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from ..programs.splits import (
    SplitSpec,
    build_split_manifest,
    default_split_specs_v0_2,
)


DATASET_VERSION_V0_2 = "spider-programs-v0.2"
SEALED_SPLIT_V0_2 = "test_sealed_v0_2"


@dataclass(frozen=True, slots=True)
class SealedEvaluationAuthorization:
    """Verified inputs for the one permitted v0.2 sealed evaluation."""

    experiment_id: str
    config_path: Path
    checkpoint_path: Path
    evidence_threshold: float
    dataset_split_digest: str
    sealed_split_sha256: str
    finalist_manifest_sha256: str
    renderer_seed: int
    row_seed_offset: int
    permuted_row_seed_offset: int
    invariance_sample_limit: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repository_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"finalist manifest has no valid {field}")
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def authorize_v0_2_sealed_evaluation(
    finalist_manifest_path: str | Path,
    split_index_path: str | Path,
    *,
    access_marker_path: str | Path,
    output_path: str | Path,
    allow_sealed: bool = False,
    repository_root: str | Path | None = None,
) -> SealedEvaluationAuthorization:
    """Validate the frozen finalist before any sealed cases are materialised.

    The caller must create ``access_marker_path`` atomically before generating
    cases. A pre-existing access marker or output makes the authorization
    permanently unavailable.
    """

    if not allow_sealed:
        raise PermissionError(
            "v0.2 sealed evaluation requires explicit --allow-v0-2-sealed"
        )
    marker = Path(access_marker_path)
    output = Path(output_path)
    if marker.exists() or output.exists():
        raise FileExistsError(
            "v0.2 sealed set has already been opened or evaluated"
        )

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    finalist_path = Path(finalist_manifest_path).resolve()
    index_path = Path(split_index_path).resolve()
    finalist = json.loads(finalist_path.read_text())
    split_index = json.loads(index_path.read_text())

    if finalist.get("sealed_opened") is not False:
        raise ValueError("finalist manifest must freeze sealed_opened=false")
    if finalist.get("dataset_version") != DATASET_VERSION_V0_2:
        raise ValueError("finalist does not use spider-programs-v0.2")
    if split_index.get("generator_version") != DATASET_VERSION_V0_2:
        raise ValueError("split index does not use spider-programs-v0.2")
    if split_index.get("sealed_cases_materialised") is not False:
        raise ValueError("split index says sealed cases were already materialised")

    dataset_digest = finalist.get("dataset_split_digest")
    if (
        not isinstance(dataset_digest, str)
        or dataset_digest != split_index.get("aggregate_sha256")
    ):
        raise ValueError("finalist and split-index aggregate hashes differ")

    sealed_specs = [
        spec
        for spec in default_split_specs_v0_2()
        if spec.name == SEALED_SPLIT_V0_2 and spec.sealed
    ]
    if len(sealed_specs) != 1:
        raise RuntimeError("v0.2 sealed split definition is not unique")
    expected_split_hash = build_split_manifest(sealed_specs[0]).sha256
    indexed_split_hash = split_index.get("split_hashes", {}).get(
        SEALED_SPLIT_V0_2
    )
    if (
        finalist.get("sealed_split_sha256") != expected_split_hash
        or indexed_split_hash != expected_split_hash
    ):
        raise ValueError("sealed split hash does not match the frozen generator")

    config_path = _resolve_repository_path(
        root,
        finalist.get("config_path"),
        "config_path",
    )
    checkpoint_path = _resolve_repository_path(
        root,
        finalist.get("checkpoint_path"),
        "checkpoint_path",
    )
    expected_artifact_root = (root / "artifacts" / "spider_v0_1").resolve()
    if not config_path.is_relative_to(expected_artifact_root):
        raise PermissionError("finalist config must be a Spider v0.1 artifact")
    if not checkpoint_path.is_relative_to(expected_artifact_root):
        raise PermissionError(
            "finalist checkpoint must be a Spider v0.1 artifact"
        )
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("frozen finalist config or checkpoint is missing")
    if _sha256(config_path) != finalist.get("config_sha256"):
        raise ValueError("frozen finalist config hash mismatch")
    if _sha256(checkpoint_path) != finalist.get("checkpoint_sha256"):
        raise ValueError("frozen finalist checkpoint hash mismatch")

    config = json.loads(config_path.read_text())
    if config.get("name") != finalist.get("experiment_id"):
        raise ValueError("finalist experiment and config names differ")
    if config.get("dataset", {}).get("version") != DATASET_VERSION_V0_2:
        raise ValueError("finalist config does not use spider-programs-v0.2")

    threshold = float(finalist.get("evidence_threshold", -1.0))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("frozen evidence threshold must be in [0, 1]")
    protocol = finalist.get("sealed_evaluation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("finalist has no sealed evaluation protocol")
    if protocol.get("calibration_split") != "validation_id":
        raise ValueError("sealed threshold must come from validation_id")
    if protocol.get("calibration_during_evaluation") is not False:
        raise ValueError("sealed evaluation must not recalibrate")

    return SealedEvaluationAuthorization(
        experiment_id=str(finalist["experiment_id"]),
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        evidence_threshold=threshold,
        dataset_split_digest=dataset_digest,
        sealed_split_sha256=expected_split_hash,
        finalist_manifest_sha256=_sha256(finalist_path),
        renderer_seed=int(protocol["renderer_seed"]),
        row_seed_offset=int(protocol["row_seed_offset"]),
        permuted_row_seed_offset=int(protocol["permuted_row_seed_offset"]),
        invariance_sample_limit=int(protocol["invariance_sample_limit"]),
    )


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
