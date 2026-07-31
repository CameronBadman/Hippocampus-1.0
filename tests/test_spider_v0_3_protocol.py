from __future__ import annotations

from hippocampus.programs import (
    default_split_specs_v0_2,
    generate_split_cases,
    make_equivalent_view,
)
from hippocampus.spider import (
    EVIDENCE_DEVELOPMENT_PROTOCOL,
    build_grouped_development_cases,
    verify_grouped_development_manifest,
)


def _cases():
    specs = {
        spec.name: spec
        for spec in default_split_specs_v0_2(case_scale=0.0625)
    }
    train = generate_split_cases(specs["train"])[:8]
    source = generate_split_cases(specs["validation_id"])[:4]
    validation = tuple(
        view
        for index, case in enumerate(source)
        for view in (
            case,
            make_equivalent_view(case, seed=9000 + index),
        )
    )
    return train, validation


def test_development_roles_keep_equivalent_views_together() -> None:
    train, validation = _cases()
    grouped = build_grouped_development_cases(
        train,
        validation,
        calibration_case_count=4,
    )

    calibration_bases = {
        case.base_case_id for case in grouped.calibration
    }
    evaluation_bases = {
        case.base_case_id for case in grouped.evaluation
    }
    assert len(grouped.calibration) == 4
    assert len(grouped.evaluation) == 4
    assert calibration_bases.isdisjoint(evaluation_bases)
    assert grouped.manifest.protocol == EVIDENCE_DEVELOPMENT_PROTOCOL
    assert not grouped.manifest.sealed_access_allowed
    verify_grouped_development_manifest(grouped)


def test_grouped_development_hash_is_deterministic_and_role_sensitive() -> None:
    train, validation = _cases()
    first = build_grouped_development_cases(
        train,
        validation,
        calibration_case_count=4,
    )
    repeated = build_grouped_development_cases(
        train,
        validation,
        calibration_case_count=4,
    )
    different_role_size = build_grouped_development_cases(
        train,
        validation,
        calibration_case_count=2,
    )

    assert first.manifest == repeated.manifest
    assert (
        first.manifest.aggregate_sha256
        != different_role_size.manifest.aggregate_sha256
    )
