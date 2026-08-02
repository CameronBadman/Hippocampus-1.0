from __future__ import annotations

from hippocampus.programs import (
    audit_aligned_program_labels,
    default_aligned_dev_specs,
    generate_aligned_dev_cases,
)
from hippocampus.spider import (
    EvidenceCandidateObservation,
    EvidencePipelineCaseReport,
    EvidenceRequirement,
    audit_frozen_evidence_policies,
)


def _candidate(
    destination: int,
    logit: float,
    *,
    required: bool,
    selected: bool,
) -> EvidenceCandidateObservation:
    return EvidenceCandidateObservation(
        round_index=0,
        arc_id=destination,
        edge_id=destination,
        source_node=0,
        destination_node=destination,
        logit=logit,
        pre_context_logit=logit,
        required=required,
        selected=selected,
        recorded=selected,
        frontier_selected=False,
    )


def _report(
    case_id: str,
    *,
    requirements: tuple[EvidenceRequirement, ...],
    candidates: tuple[EvidenceCandidateObservation, ...],
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> EvidencePipelineCaseReport:
    return EvidencePipelineCaseReport(
        case_id=case_id,
        family="lookup",
        horizon=1,
        requirements=requirements,
        requirement_observations=(),
        candidate_observations=candidates,
        exact_set_accuracy=float(false_positives == 0 and false_negatives == 0),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        predicted_cardinality=true_positives + false_positives,
        required_cardinality=len(requirements),
        average_precision=1.0,
        worst_positive_rank=1 if requirements else None,
        minimum_positive_negative_margin=1.0 if requirements else None,
    )


def test_frozen_policy_audit_separates_ranking_from_global_threshold() -> None:
    positive = _report(
        "positive",
        requirements=(EvidenceRequirement(None, None, 1),),
        candidates=(
            _candidate(1, 2.0, required=True, selected=False),
            _candidate(2, 1.0, required=False, selected=False),
        ),
        true_positives=0,
        false_positives=0,
        false_negatives=1,
    )
    empty = _report(
        "empty",
        requirements=(),
        candidates=(_candidate(3, 0.5, required=False, selected=True),),
        true_positives=0,
        false_positives=1,
        false_negatives=0,
    )

    audit = audit_frozen_evidence_policies((positive, empty))

    assert audit.overall["P0_global_threshold"].exact_set_accuracy == 0.0
    assert audit.overall["P1_oracle_cardinality"].exact_set_accuracy == 1.0
    assert audit.overall["P2_per_case_threshold"].exact_set_accuracy == 1.0
    assert audit.overall["P3_oracle_null"].exact_set_accuracy == 0.5
    assert audit.oracle_cardinality_exact_set_gain == 1.0
    assert audit.recommended_branch == "set_decoding"


def test_aligned_generator_labels_are_mechanically_consistent() -> None:
    cases = generate_aligned_dev_cases(
        default_aligned_dev_specs()[0],
        limit=256,
    )

    report = audit_aligned_program_labels(cases)

    assert report.case_count == 256
    assert report.invalid_case_count == 0
    assert report.evidence_label_mismatch_count == 0
    assert report.positive_summary_identity_rate == 1.0
    assert report.lookup_observable_rule_accuracy == 1.0
    assert report.unsupported_symbol_reuse_count == 0
