from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .evidence_diagnostics import (
    EvidenceCandidateObservation,
    EvidencePipelineCaseReport,
)
from .state_oracle import EvidenceRequirement


@dataclass(frozen=True, slots=True)
class FrozenEvidencePolicyMetrics:
    case_count: int
    exact_set_accuracy: float
    precision: float
    recall: float
    false_positives_per_case: float
    mean_predicted_cardinality: float
    mean_required_cardinality: float
    mean_absolute_cardinality_error: float


@dataclass(frozen=True, slots=True)
class FrozenEvidencePolicyAudit:
    overall: dict[str, FrozenEvidencePolicyMetrics]
    by_family: dict[str, dict[str, FrozenEvidencePolicyMetrics]]
    oracle_cardinality_exact_set_gain: float
    recommended_branch: str

    def as_dict(self) -> dict[str, object]:
        return {
            "overall": {
                name: asdict(metrics)
                for name, metrics in self.overall.items()
            },
            "by_family": {
                family: {
                    name: asdict(metrics)
                    for name, metrics in policies.items()
                }
                for family, policies in self.by_family.items()
            },
            "oracle_cardinality_exact_set_gain": (
                self.oracle_cardinality_exact_set_gain
            ),
            "recommended_branch": self.recommended_branch,
        }


def evidence_pipeline_case_report_from_dict(
    value: dict[str, Any],
) -> EvidencePipelineCaseReport:
    """Restore the immutable candidate observations stored in an evaluation."""

    requirements = tuple(
        EvidenceRequirement(**item) for item in value["requirements"]
    )
    candidates = tuple(
        EvidenceCandidateObservation(**item)
        for item in value["candidate_observations"]
    )
    return EvidencePipelineCaseReport(
        case_id=str(value["case_id"]),
        family=str(value["family"]),
        horizon=int(value["horizon"]),
        requirements=requirements,
        requirement_observations=(),
        candidate_observations=candidates,
        exact_set_accuracy=float(value["exact_set_accuracy"]),
        true_positives=int(value["true_positives"]),
        false_positives=int(value["false_positives"]),
        false_negatives=int(value["false_negatives"]),
        predicted_cardinality=int(value["predicted_cardinality"]),
        required_cardinality=int(value["required_cardinality"]),
        average_precision=float(value["average_precision"]),
        worst_positive_rank=(
            None
            if value["worst_positive_rank"] is None
            else int(value["worst_positive_rank"])
        ),
        minimum_positive_negative_margin=(
            None
            if value["minimum_positive_negative_margin"] is None
            else float(value["minimum_positive_negative_margin"])
        ),
    )


def _requirement_key(requirement, *, edge_specific: bool):
    if not edge_specific:
        return (None, None, requirement.destination_node)
    return (
        requirement.edge_id,
        requirement.source_node,
        requirement.destination_node,
    )


def _candidate_key(candidate, *, edge_specific: bool):
    if not edge_specific:
        return (None, None, candidate.destination_node)
    return (candidate.edge_id, candidate.source_node, candidate.destination_node)


def _sets_for_report(report: EvidencePipelineCaseReport):
    edge_specific = any(
        requirement.edge_specific for requirement in report.requirements
    )
    required = {
        _requirement_key(requirement, edge_specific=edge_specific)
        for requirement in report.requirements
    }
    scores: dict[tuple[object, ...], float] = {}
    selected: set[tuple[object, ...]] = set()
    for candidate in report.candidate_observations:
        key = _candidate_key(candidate, edge_specific=edge_specific)
        scores[key] = max(scores.get(key, float("-inf")), candidate.logit)
        if candidate.selected:
            selected.add(key)
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    oracle_cardinality = set(ranked[: len(required)])
    oracle_null = selected if required else set()

    threshold_candidates: list[set[tuple[object, ...]]] = [set()]
    current: set[tuple[object, ...]] = set()
    previous_score: float | None = None
    for key in ranked:
        score = scores[key]
        if previous_score is not None and score != previous_score:
            threshold_candidates.append(set(current))
        current.add(key)
        previous_score = score
    threshold_candidates.append(set(current))

    def threshold_key(predicted: set[tuple[object, ...]]):
        true_positive = len(predicted & required)
        false_positive = len(predicted - required)
        false_negative = len(required - predicted)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return (
            float(predicted == required),
            f1,
            recall,
            precision,
            -false_positive,
            -len(predicted),
        )

    per_case_threshold = max(threshold_candidates, key=threshold_key)
    return required, oracle_cardinality, per_case_threshold, oracle_null


def _metrics_from_sets(
    rows: Sequence[
        tuple[set[tuple[object, ...]], set[tuple[object, ...]]]
    ],
) -> FrozenEvidencePolicyMetrics:
    true_positive = sum(len(predicted & required) for required, predicted in rows)
    false_positive = sum(len(predicted - required) for required, predicted in rows)
    false_negative = sum(len(required - predicted) for required, predicted in rows)
    return FrozenEvidencePolicyMetrics(
        case_count=len(rows),
        exact_set_accuracy=(
            sum(predicted == required for required, predicted in rows)
            / max(1, len(rows))
        ),
        precision=true_positive / max(1, true_positive + false_positive),
        recall=true_positive / max(1, true_positive + false_negative),
        false_positives_per_case=false_positive / max(1, len(rows)),
        mean_predicted_cardinality=(
            sum(len(predicted) for _, predicted in rows) / max(1, len(rows))
        ),
        mean_required_cardinality=(
            sum(len(required) for required, _ in rows) / max(1, len(rows))
        ),
        mean_absolute_cardinality_error=(
            sum(
                abs(len(predicted) - len(required))
                for required, predicted in rows
            )
            / max(1, len(rows))
        ),
    )


def _p0_metrics(
    reports: Sequence[EvidencePipelineCaseReport],
) -> FrozenEvidencePolicyMetrics:
    true_positive = sum(report.true_positives for report in reports)
    false_positive = sum(report.false_positives for report in reports)
    false_negative = sum(report.false_negatives for report in reports)
    return FrozenEvidencePolicyMetrics(
        case_count=len(reports),
        exact_set_accuracy=(
            sum(report.exact_set_accuracy for report in reports)
            / max(1, len(reports))
        ),
        precision=true_positive / max(1, true_positive + false_positive),
        recall=true_positive / max(1, true_positive + false_negative),
        false_positives_per_case=false_positive / max(1, len(reports)),
        mean_predicted_cardinality=(
            sum(report.predicted_cardinality for report in reports)
            / max(1, len(reports))
        ),
        mean_required_cardinality=(
            sum(report.required_cardinality for report in reports)
            / max(1, len(reports))
        ),
        mean_absolute_cardinality_error=(
            sum(
                abs(report.predicted_cardinality - report.required_cardinality)
                for report in reports
            )
            / max(1, len(reports))
        ),
    )


def _audit_group(
    reports: Sequence[EvidencePipelineCaseReport],
) -> dict[str, FrozenEvidencePolicyMetrics]:
    resolved = [_sets_for_report(report) for report in reports]
    return {
        "P0_global_threshold": _p0_metrics(reports),
        "P1_oracle_cardinality": _metrics_from_sets(
            [(required, oracle_k) for required, oracle_k, _, _ in resolved]
        ),
        "P2_per_case_threshold": _metrics_from_sets(
            [(required, threshold) for required, _, threshold, _ in resolved]
        ),
        "P3_oracle_null": _metrics_from_sets(
            [(required, oracle_null) for required, _, _, oracle_null in resolved]
        ),
    }


def audit_frozen_evidence_policies(
    reports: Sequence[EvidencePipelineCaseReport],
) -> FrozenEvidencePolicyAudit:
    """Measure ranking and set-decoding ceilings without another model run."""

    if not reports:
        raise ValueError("frozen evidence policy audit requires case reports")
    overall = _audit_group(reports)
    families = sorted({report.family for report in reports})
    by_family = {
        family: _audit_group(
            [report for report in reports if report.family == family]
        )
        for family in families
    }
    gain = (
        overall["P1_oracle_cardinality"].exact_set_accuracy
        - overall["P0_global_threshold"].exact_set_accuracy
    )
    return FrozenEvidencePolicyAudit(
        overall=overall,
        by_family=by_family,
        oracle_cardinality_exact_set_gain=gain,
        recommended_branch=(
            "set_decoding" if gain >= 0.15 else "ranking_and_hard_negatives"
        ),
    )
