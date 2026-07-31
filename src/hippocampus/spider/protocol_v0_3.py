from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Sequence

from ..programs.schema import GraphProgramCase


EVIDENCE_DEVELOPMENT_PROTOCOL = "spider-v0.3-evidence-dev"


def _logical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DevelopmentPartitionManifest:
    name: str
    case_ids: tuple[str, ...]
    base_case_ids: tuple[str, ...]
    family_counts: dict[str, int]
    termination_counts: dict[str, int]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroupedDevelopmentManifest:
    protocol: str
    source_dataset_version: str
    train: DevelopmentPartitionManifest
    calibration: DevelopmentPartitionManifest
    evaluation: DevelopmentPartitionManifest
    aggregate_sha256: str
    sealed_access_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroupedDevelopmentCases:
    train: tuple[GraphProgramCase, ...]
    calibration: tuple[GraphProgramCase, ...]
    evaluation: tuple[GraphProgramCase, ...]
    manifest: GroupedDevelopmentManifest


def _partition_manifest(
    name: str,
    cases: Sequence[GraphProgramCase],
) -> DevelopmentPartitionManifest:
    family_counts: dict[str, int] = {}
    termination_counts: dict[str, int] = {}
    for case in cases:
        family_counts[case.family.value] = (
            family_counts.get(case.family.value, 0) + 1
        )
        decision = case.termination.decision.value
        termination_counts[decision] = termination_counts.get(decision, 0) + 1
    case_ids = tuple(case.case_id for case in cases)
    base_case_ids = tuple(sorted({case.base_case_id for case in cases}))
    payload = {
        "protocol": EVIDENCE_DEVELOPMENT_PROTOCOL,
        "name": name,
        "case_ids": case_ids,
        "base_case_ids": base_case_ids,
        "family_counts": family_counts,
        "termination_counts": termination_counts,
    }
    return DevelopmentPartitionManifest(
        name=name,
        case_ids=case_ids,
        base_case_ids=base_case_ids,
        family_counts=family_counts,
        termination_counts=termination_counts,
        sha256=_logical_hash(payload),
    )


def _calibration_base_ids(
    cases: Sequence[GraphProgramCase],
    *,
    target_case_count: int,
    salt: str,
) -> set[str]:
    groups: dict[str, list[GraphProgramCase]] = {}
    for case in cases:
        groups.setdefault(case.base_case_id, []).append(case)
    ordered_groups = sorted(
        groups,
        key=lambda base_case_id: hashlib.sha256(
            f"{salt}|{base_case_id}".encode()
        ).hexdigest(),
    )

    # Exact subset sum keeps aligned views together. For future datasets with
    # larger view groups, the largest attainable count below the target is
    # selected instead of splitting a latent case across roles.
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for base_case_id in ordered_groups:
        size = len(groups[base_case_id])
        additions = {
            count + size: (*selected, base_case_id)
            for count, selected in tuple(choices.items())
            if count + size <= target_case_count
        }
        for count, selected in additions.items():
            choices.setdefault(count, selected)
    selected_count = max(choices)
    if selected_count == 0:
        raise ValueError("calibration partition would be empty")
    return set(choices[selected_count])


def build_grouped_development_cases(
    train_cases: Sequence[GraphProgramCase],
    validation_cases: Sequence[GraphProgramCase],
    *,
    calibration_case_count: int = 64,
    source_dataset_version: str = "spider-programs-v0.2",
    salt: str = EVIDENCE_DEVELOPMENT_PROTOCOL,
) -> GroupedDevelopmentCases:
    """Split development cases by latent base ID, never by rendered view."""

    if not train_cases or not validation_cases:
        raise ValueError("train and validation cases may not be empty")
    if not 0 < calibration_case_count < len(validation_cases):
        raise ValueError(
            "calibration_case_count must leave a non-empty evaluation set"
        )
    all_case_ids = [
        case.case_id for case in (*train_cases, *validation_cases)
    ]
    if len(set(all_case_ids)) != len(all_case_ids):
        raise ValueError("development case IDs must be globally unique")

    calibration_bases = _calibration_base_ids(
        validation_cases,
        target_case_count=calibration_case_count,
        salt=salt,
    )
    calibration = tuple(
        case
        for case in validation_cases
        if case.base_case_id in calibration_bases
    )
    evaluation = tuple(
        case
        for case in validation_cases
        if case.base_case_id not in calibration_bases
    )
    calibration_base_ids = {case.base_case_id for case in calibration}
    evaluation_base_ids = {case.base_case_id for case in evaluation}
    if calibration_base_ids & evaluation_base_ids:
        raise AssertionError("base-case leakage across development roles")
    if not evaluation:
        raise ValueError("evaluation partition would be empty")

    train_manifest = _partition_manifest("train", train_cases)
    calibration_manifest = _partition_manifest(
        "development_calibration",
        calibration,
    )
    evaluation_manifest = _partition_manifest(
        "development_evaluation",
        evaluation,
    )
    aggregate = _logical_hash(
        {
            "protocol": EVIDENCE_DEVELOPMENT_PROTOCOL,
            "source_dataset_version": source_dataset_version,
            "partitions": {
                "train": train_manifest.sha256,
                "calibration": calibration_manifest.sha256,
                "evaluation": evaluation_manifest.sha256,
            },
            "sealed_access_allowed": False,
        }
    )
    manifest = GroupedDevelopmentManifest(
        protocol=EVIDENCE_DEVELOPMENT_PROTOCOL,
        source_dataset_version=source_dataset_version,
        train=train_manifest,
        calibration=calibration_manifest,
        evaluation=evaluation_manifest,
        aggregate_sha256=aggregate,
    )
    return GroupedDevelopmentCases(
        train=tuple(train_cases),
        calibration=calibration,
        evaluation=evaluation,
        manifest=manifest,
    )


def verify_grouped_development_manifest(
    cases: GroupedDevelopmentCases,
) -> None:
    calibration_bases = {
        case.base_case_id for case in cases.calibration
    }
    evaluation_bases = {case.base_case_id for case in cases.evaluation}
    if calibration_bases & evaluation_bases:
        raise ValueError("base-case leakage across development roles")
    train = _partition_manifest("train", cases.train)
    calibration = _partition_manifest(
        "development_calibration",
        cases.calibration,
    )
    evaluation = _partition_manifest(
        "development_evaluation",
        cases.evaluation,
    )
    aggregate = _logical_hash(
        {
            "protocol": EVIDENCE_DEVELOPMENT_PROTOCOL,
            "source_dataset_version": (
                cases.manifest.source_dataset_version
            ),
            "partitions": {
                "train": train.sha256,
                "calibration": calibration.sha256,
                "evaluation": evaluation.sha256,
            },
            "sealed_access_allowed": False,
        }
    )
    rebuilt = GroupedDevelopmentManifest(
        protocol=EVIDENCE_DEVELOPMENT_PROTOCOL,
        source_dataset_version=(
            cases.manifest.source_dataset_version
        ),
        train=train,
        calibration=calibration,
        evaluation=evaluation,
        aggregate_sha256=aggregate,
    )
    if rebuilt != cases.manifest:
        raise ValueError("grouped development manifest is not reproducible")
