from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from typing import Sequence

import torch
from torch.nn import functional as F

from ..programs.batching import PackedProgramBatch
from .calibration import validate_calibration_source
from .config import SparseControllerConfig
from .evaluation_v0_1 import execute_closed_loop_case
from .execution import ControllerExecutionPolicy
from .model import CandidateScorerBase


@dataclass(frozen=True, slots=True)
class TemperatureScalingResult:
    fitted_temperature: float
    applied_temperature: float
    accepted: bool
    baseline_ece: float
    fitted_ece: float
    baseline_nll: float
    fitted_nll: float
    baseline_brier: float
    fitted_brier: float


@dataclass(frozen=True, slots=True)
class ExactSetOperatingPoint:
    calibrated_probability_threshold: float
    raw_probability_threshold: float
    exact_set_accuracy: float
    precision: float
    recall: float
    false_positives_per_case: float
    mean_predicted_cardinality: float
    scored_positive_coverage: float


@dataclass(frozen=True, slots=True)
class ClosedLoopEvidenceCalibration:
    split_name: str
    dataset_version: str
    source_case_hash: str
    source_case_count: int
    precision_floor: float
    constraint_satisfied: bool
    selected: ExactSetOperatingPoint
    temperature: TemperatureScalingResult
    curve: tuple[ExactSetOperatingPoint, ...]

    @property
    def threshold(self) -> float:
        return self.selected.raw_probability_threshold

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _binary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[float, float, float]:
    values = logits.detach().float().flatten().cpu() / temperature
    targets = labels.detach().float().flatten().cpu()
    if values.numel() == 0:
        return 0.0, 0.0, 0.0
    probabilities = torch.sigmoid(values)
    nll = float(
        F.binary_cross_entropy_with_logits(values, targets).item()
    )
    brier = float(((probabilities - targets) ** 2).mean().item())
    ece = 0.0
    for lower in torch.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (
            (probabilities >= lower)
            & (
                probabilities <= upper
                if float(upper) >= 1.0
                else probabilities < upper
            )
        )
        if not bool(mask.any().item()):
            continue
        confidence = probabilities[mask].mean()
        frequency = targets[mask].mean()
        ece += float(
            mask.float().mean().item()
            * abs(float(confidence.item() - frequency.item()))
        )
    return nll, brier, ece


def fit_temperature_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    baseline_ece_threshold: float = 0.05,
    minimum_nll_improvement: float = 0.0001,
) -> TemperatureScalingResult:
    """Fit one deterministic scalar and apply it only under registered gates."""

    values = logits.detach().float().flatten().cpu()
    targets = labels.detach().bool().flatten().cpu()
    if values.numel() != targets.numel() or values.numel() == 0:
        raise ValueError("temperature fitting requires aligned observations")
    baseline_nll, baseline_brier, baseline_ece = _binary_metrics(
        values,
        targets,
        temperature=1.0,
    )
    candidates = torch.exp(
        torch.linspace(math.log(0.25), math.log(4.0), 121)
    ).tolist()
    fitted_temperature = min(
        candidates,
        key=lambda value: _binary_metrics(
            values,
            targets,
            temperature=float(value),
        )[0],
    )
    fitted_nll, fitted_brier, fitted_ece = _binary_metrics(
        values,
        targets,
        temperature=float(fitted_temperature),
    )
    accepted = (
        baseline_ece > baseline_ece_threshold
        and baseline_nll - fitted_nll >= minimum_nll_improvement
        and fitted_brier <= baseline_brier
    )
    return TemperatureScalingResult(
        fitted_temperature=float(fitted_temperature),
        applied_temperature=(
            float(fitted_temperature) if accepted else 1.0
        ),
        accepted=accepted,
        baseline_ece=baseline_ece,
        fitted_ece=fitted_ece,
        baseline_nll=baseline_nll,
        fitted_nll=fitted_nll,
        baseline_brier=baseline_brier,
        fitted_brier=fitted_brier,
    )


def _raw_probability_threshold(
    calibrated_probability: float,
    temperature: float,
) -> float:
    clipped = min(max(calibrated_probability, 1e-6), 1.0 - 1e-6)
    logit = math.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + math.exp(-temperature * logit))


def _evaluate_threshold(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    controller_config: SparseControllerConfig,
    execution_policy: ControllerExecutionPolicy | None,
    calibrated_threshold: float,
    raw_threshold: float,
) -> ExactSetOperatingPoint:
    resolved = replace(
        controller_config,
        evidence_threshold=raw_threshold,
    )
    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact = 0.0
    predicted = 0
    reachable = 0
    scored = 0
    for batch in batches:
        execution = execute_closed_loop_case(
            model,
            batch,
            resolved,
            execution_policy,
        )
        report = execution.evidence_pipeline
        true_positive += report.true_positives
        false_positive += report.false_positives
        false_negative += report.false_negatives
        exact += report.exact_set_accuracy
        predicted += report.predicted_cardinality
        for requirement in report.requirements:
            observations = [
                row
                for row in report.requirement_observations
                if row.requirement == requirement
                and row.outstanding_before
            ]
            if any(row.reachable for row in observations):
                reachable += 1
                scored += int(any(row.scored for row in observations))
    return ExactSetOperatingPoint(
        calibrated_probability_threshold=calibrated_threshold,
        raw_probability_threshold=raw_threshold,
        exact_set_accuracy=exact / len(batches),
        precision=true_positive / max(1, true_positive + false_positive),
        recall=true_positive / max(1, true_positive + false_negative),
        false_positives_per_case=false_positive / len(batches),
        mean_predicted_cardinality=predicted / len(batches),
        scored_positive_coverage=scored / max(1, reachable),
    )


def calibrate_closed_loop_evidence(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    controller_config: SparseControllerConfig,
    split_name: str = "development_calibration",
    dataset_version: str = "spider-programs-v0.2",
    precision_floor: float = 0.0,
    execution_policy: ControllerExecutionPolicy | None = None,
    coarse_thresholds: Sequence[float] | None = None,
    fine_radius: float = 0.05,
    fine_step: float = 0.005,
) -> ClosedLoopEvidenceCalibration:
    """Choose an exact-set operating point on development calibration only."""

    validate_calibration_source(
        split_name=split_name,
        dataset_version=dataset_version,
    )
    if not batches:
        raise ValueError("closed-loop calibration requires development cases")
    if not 0.0 <= precision_floor <= 1.0:
        raise ValueError("precision_floor must be in [0, 1]")
    if fine_radius < 0 or fine_step <= 0:
        raise ValueError("fine calibration settings are invalid")

    was_training = model.training
    model.eval()
    reference_scores: list[torch.Tensor] = []
    reference_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            execution = execute_closed_loop_case(
                model,
                batch,
                controller_config,
                execution_policy,
            )
            reference_scores.append(execution.candidate_evidence_logits)
            reference_labels.append(execution.candidate_evidence_labels)
        temperature = fit_temperature_scaling(
            torch.cat(reference_scores),
            torch.cat(reference_labels),
        )
        coarse = tuple(
            coarse_thresholds
            if coarse_thresholds is not None
            else (index / 20 for index in range(1, 20))
        )
        if not coarse or any(
            threshold <= 0 or threshold >= 1 for threshold in coarse
        ):
            raise ValueError("coarse thresholds must lie strictly in (0, 1)")
        cache: dict[float, ExactSetOperatingPoint] = {}

        def evaluate(calibrated_threshold: float) -> ExactSetOperatingPoint:
            calibrated = round(calibrated_threshold, 12)
            if calibrated not in cache:
                raw = _raw_probability_threshold(
                    calibrated,
                    temperature.applied_temperature,
                )
                cache[calibrated] = _evaluate_threshold(
                    model,
                    batches,
                    controller_config=controller_config,
                    execution_policy=execution_policy,
                    calibrated_threshold=calibrated,
                    raw_threshold=raw,
                )
            return cache[calibrated]

        coarse_points = [evaluate(value) for value in coarse]

        def selection_key(
            point: ExactSetOperatingPoint,
        ) -> tuple[float, float, float, float, float]:
            return (
                point.exact_set_accuracy,
                point.recall,
                point.precision,
                -point.false_positives_per_case,
                point.calibrated_probability_threshold,
            )

        eligible_coarse = [
            point
            for point in coarse_points
            if point.precision >= precision_floor
        ]
        coarse_winner = max(
            eligible_coarse or coarse_points,
            key=selection_key,
        )
        fine_values: list[float] = []
        if fine_radius > 0:
            steps = round((2 * fine_radius) / fine_step)
            fine_values = [
                min(
                    0.999,
                    max(
                        0.001,
                        coarse_winner.calibrated_probability_threshold
                        - fine_radius
                        + index * fine_step,
                    ),
                )
                for index in range(steps + 1)
            ]
        for value in fine_values:
            evaluate(value)
        curve = tuple(
            cache[key] for key in sorted(cache)
        )
        eligible = [
            point for point in curve if point.precision >= precision_floor
        ]
        selected = max(eligible or curve, key=selection_key)
    model.train(was_training)

    source_case_ids = tuple(batch.cases[0].case_id for batch in batches)
    source_hash = hashlib.sha256(
        "\n".join(source_case_ids).encode()
    ).hexdigest()
    return ClosedLoopEvidenceCalibration(
        split_name=split_name,
        dataset_version=dataset_version,
        source_case_hash=source_hash,
        source_case_count=len(batches),
        precision_floor=precision_floor,
        constraint_satisfied=bool(eligible),
        selected=selected,
        temperature=temperature,
        curve=curve,
    )
