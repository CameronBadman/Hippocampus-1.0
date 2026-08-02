from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from ..programs.batching import PackedProgramBatch
from .calibration import validate_calibration_source
from .calibration_v0_3 import (
    ClosedLoopEvidenceCalibration,
    ExactSetOperatingPoint,
    TemperatureScalingResult,
    _binary_metrics,
    _evaluate_threshold,
    _raw_probability_threshold,
    fit_temperature_scaling,
)
from .config import SparseControllerConfig
from .evidence_diagnostics import (
    EvidencePipelineCaseReport,
    aggregate_evidence_pipeline,
)
from .evaluation_v0_1 import execute_closed_loop_case
from .execution import ControllerExecutionPolicy
from .model import CandidateScorerBase


@dataclass(frozen=True, slots=True)
class FastEvidenceCalibrationResult:
    """Exact shortlisted calibration plus its frozen-logit audit trail."""

    calibration: ClosedLoopEvidenceCalibration
    reference_threshold: float
    reference_pipeline: dict[str, object]
    approximate_curve: tuple[ExactSetOperatingPoint, ...]
    exact_candidate_thresholds: tuple[float, ...]
    temperature_fitted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration": self.calibration.as_dict(),
            "reference_threshold": self.reference_threshold,
            "reference_pipeline": self.reference_pipeline,
            "approximate_curve": [
                asdict(point) for point in self.approximate_curve
            ],
            "exact_candidate_thresholds": list(
                self.exact_candidate_thresholds
            ),
            "temperature_fitted": self.temperature_fitted,
        }


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


def approximate_exact_set_point(
    reports: Sequence[EvidencePipelineCaseReport],
    *,
    calibrated_threshold: float,
    temperature: float,
    evidence_budget: int,
) -> ExactSetOperatingPoint:
    """Evaluate a global threshold against frozen candidate trajectories.

    This cheap stage only shortlists thresholds. Final operating points are
    always rerun through the real recurrent controller so evidence-state
    feedback and budget arithmetic remain authoritative.
    """

    if not reports:
        raise ValueError("approximate calibration requires case reports")
    raw_threshold = _raw_probability_threshold(
        calibrated_threshold,
        temperature,
    )
    logit_threshold = math.log(raw_threshold / (1.0 - raw_threshold))
    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact = 0
    predicted_count = 0
    reachable = 0
    scored = 0
    for report in reports:
        edge_specific = any(
            requirement.edge_specific for requirement in report.requirements
        )
        required = {
            _requirement_key(requirement, edge_specific=edge_specific)
            for requirement in report.requirements
        }
        best_by_key: dict[tuple[object, ...], float] = {}
        for candidate in report.candidate_observations:
            key = _candidate_key(candidate, edge_specific=edge_specific)
            best_by_key[key] = max(
                best_by_key.get(key, float("-inf")),
                candidate.logit,
            )
        eligible = [
            (key, logit)
            for key, logit in best_by_key.items()
            if logit >= logit_threshold
        ]
        eligible.sort(key=lambda item: (-item[1], item[0]))
        predicted = {
            key for key, _ in eligible[: max(0, evidence_budget)]
        }
        true_positive += len(predicted & required)
        false_positive += len(predicted - required)
        false_negative += len(required - predicted)
        exact += int(predicted == required)
        predicted_count += len(predicted)
        for requirement in report.requirements:
            observations = [
                row
                for row in report.requirement_observations
                if row.requirement == requirement and row.outstanding_before
            ]
            if any(row.reachable for row in observations):
                reachable += 1
                scored += int(any(row.scored for row in observations))
    return ExactSetOperatingPoint(
        calibrated_probability_threshold=calibrated_threshold,
        raw_probability_threshold=raw_threshold,
        exact_set_accuracy=exact / len(reports),
        precision=true_positive / max(1, true_positive + false_positive),
        recall=true_positive / max(1, true_positive + false_negative),
        false_positives_per_case=false_positive / len(reports),
        mean_predicted_cardinality=predicted_count / len(reports),
        scored_positive_coverage=scored / max(1, reachable),
    )


def _point_key(
    point: ExactSetOperatingPoint,
    *,
    precision_floor: float,
    coverage_floor: float,
) -> tuple[float, ...]:
    eligible = (
        point.precision >= precision_floor
        and point.scored_positive_coverage >= coverage_floor
    )
    return (
        float(eligible),
        point.exact_set_accuracy,
        point.recall,
        point.precision,
        -point.false_positives_per_case,
        -point.calibrated_probability_threshold,
    )


def _identity_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> TemperatureScalingResult:
    nll, brier, ece = _binary_metrics(logits, labels, temperature=1.0)
    return TemperatureScalingResult(
        fitted_temperature=1.0,
        applied_temperature=1.0,
        accepted=False,
        baseline_ece=ece,
        fitted_ece=ece,
        baseline_nll=nll,
        fitted_nll=nll,
        baseline_brier=brier,
        fitted_brier=brier,
    )


def fast_calibrate_closed_loop_evidence(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    controller_config: SparseControllerConfig,
    split_name: str,
    dataset_version: str,
    precision_floor: float = 0.90,
    coverage_floor: float = 0.98,
    execution_policy: ControllerExecutionPolicy | None = None,
    fit_temperature: bool = True,
    reference_threshold: float = 0.5,
    approximate_thresholds: Sequence[float] | None = None,
    exact_candidate_count: int = 7,
) -> FastEvidenceCalibrationResult:
    """Calibrate efficiently while preserving an exact controller decision.

    One reference rollout produces a dense frozen-logit sweep. Only the best
    preregistered threshold candidates are rerun through the full controller.
    This bounds calibration cost without treating the frozen approximation as
    the deployable result.
    """

    validate_calibration_source(
        split_name=split_name,
        dataset_version=dataset_version,
    )
    if not batches:
        raise ValueError("calibration requires development batches")
    if not 0 <= precision_floor <= 1 or not 0 <= coverage_floor <= 1:
        raise ValueError("precision and coverage floors must be in [0, 1]")
    if exact_candidate_count <= 0:
        raise ValueError("exact_candidate_count must be positive")
    reference_config = SparseControllerConfig(
        **{
            **asdict(controller_config),
            "evidence_threshold": reference_threshold,
        }
    )
    was_training = model.training
    model.eval()
    reports: list[EvidencePipelineCaseReport] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            execution = execute_closed_loop_case(
                model,
                batch,
                reference_config,
                execution_policy,
            )
            reports.append(execution.evidence_pipeline)
            logits.append(execution.candidate_evidence_logits)
            labels.append(execution.candidate_evidence_labels)
        all_logits = torch.cat(logits)
        all_labels = torch.cat(labels)
        temperature = (
            fit_temperature_scaling(all_logits, all_labels)
            if fit_temperature
            else _identity_temperature(all_logits, all_labels)
        )
        thresholds = tuple(
            approximate_thresholds
            if approximate_thresholds is not None
            else (index / 100 for index in range(1, 100))
        )
        if not thresholds or any(value <= 0 or value >= 1 for value in thresholds):
            raise ValueError("approximate thresholds must lie in (0, 1)")
        approximate = tuple(
            approximate_exact_set_point(
                reports,
                calibrated_threshold=float(threshold),
                temperature=temperature.applied_temperature,
                evidence_budget=controller_config.evidence_selection_budget,
            )
            for threshold in thresholds
        )
        shortlisted = sorted(
            approximate,
            key=lambda point: _point_key(
                point,
                precision_floor=precision_floor,
                coverage_floor=coverage_floor,
            ),
            reverse=True,
        )[:exact_candidate_count]
        exact_points = tuple(
            _evaluate_threshold(
                model,
                batches,
                controller_config=controller_config,
                execution_policy=execution_policy,
                calibrated_threshold=point.calibrated_probability_threshold,
                raw_threshold=point.raw_probability_threshold,
            )
            for point in shortlisted
        )
    model.train(was_training)
    selected = max(
        exact_points,
        key=lambda point: _point_key(
            point,
            precision_floor=precision_floor,
            coverage_floor=coverage_floor,
        ),
    )
    constraint_satisfied = (
        selected.precision >= precision_floor
        and selected.scored_positive_coverage >= coverage_floor
    )
    source_case_ids = tuple(batch.cases[0].case_id for batch in batches)
    calibration = ClosedLoopEvidenceCalibration(
        split_name=split_name,
        dataset_version=dataset_version,
        source_case_hash=hashlib.sha256(
            "\n".join(source_case_ids).encode()
        ).hexdigest(),
        source_case_count=len(batches),
        precision_floor=precision_floor,
        constraint_satisfied=constraint_satisfied,
        selected=selected,
        temperature=temperature,
        curve=tuple(
            sorted(
                exact_points,
                key=lambda point: point.calibrated_probability_threshold,
            )
        ),
    )
    return FastEvidenceCalibrationResult(
        calibration=calibration,
        reference_threshold=reference_threshold,
        reference_pipeline=aggregate_evidence_pipeline(
            reports,
            include_cases=False,
        ),
        approximate_curve=approximate,
        exact_candidate_thresholds=tuple(
            point.calibrated_probability_threshold for point in shortlisted
        ),
        temperature_fitted=fit_temperature,
    )
