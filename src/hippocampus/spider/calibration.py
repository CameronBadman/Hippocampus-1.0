from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True, slots=True)
class PrecisionRecallPoint:
    threshold: float
    precision: float
    recall: float


@dataclass(frozen=True, slots=True)
class EvidenceCalibration:
    split_name: str
    dataset_version: str
    threshold: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    positive_count: int
    negative_count: int
    curve: tuple[PrecisionRecallPoint, ...]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["curve"] = [asdict(point) for point in self.curve]
        return result


def _validate_binary_inputs(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_scores = scores.detach().float().flatten().cpu()
    resolved_labels = labels.detach().bool().flatten().cpu()
    if resolved_scores.numel() != resolved_labels.numel():
        raise ValueError("evidence scores and labels must align")
    if resolved_scores.numel() and not bool(
        torch.isfinite(resolved_scores).all().item()
    ):
        raise ValueError("evidence scores must be finite")
    return resolved_scores, resolved_labels


def binary_average_precision(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Threshold-free average precision with stable tie handling."""

    values, targets = _validate_binary_inputs(scores, labels)
    positives = int(targets.sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(values, descending=True, stable=True)
    ranked = targets[order].to(torch.float64)
    cumulative_true = torch.cumsum(ranked, dim=0)
    ranks = torch.arange(1, ranked.numel() + 1, dtype=torch.float64)
    precision = cumulative_true / ranks
    return float(precision[ranked.bool()].sum().item() / positives)


def precision_recall_curve(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[PrecisionRecallPoint, ...]:
    values, targets = _validate_binary_inputs(scores, labels)
    probabilities = torch.sigmoid(values)
    thresholds = sorted(
        {
            0.0,
            1.0,
            *(float(value) for value in probabilities.tolist()),
        }
    )
    positive_count = int(targets.sum().item())
    points: list[PrecisionRecallPoint] = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        true_positive = int((predicted & targets).sum().item())
        false_positive = int((predicted & ~targets).sum().item())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, positive_count)
        points.append(
            PrecisionRecallPoint(
                threshold=threshold,
                precision=precision,
                recall=recall,
            )
        )
    return tuple(points)


def calibrate_evidence_threshold(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    split_name: str,
    dataset_version: str,
) -> EvidenceCalibration:
    """Choose a deterministic evidence-F1 operating point on allowed dev data."""

    validate_calibration_source(
        split_name=split_name,
        dataset_version=dataset_version,
    )
    values, targets = _validate_binary_inputs(scores, labels)
    curve = precision_recall_curve(values, targets)
    if not curve:
        raise ValueError("calibration requires at least one scored candidate")

    def operating_key(point: PrecisionRecallPoint) -> tuple[float, float, float]:
        f1 = (
            2 * point.precision * point.recall
            / max(1e-12, point.precision + point.recall)
        )
        return f1, point.recall, -point.threshold

    selected = max(curve, key=operating_key)
    f1 = (
        2 * selected.precision * selected.recall
        / max(1e-12, selected.precision + selected.recall)
    )
    positive_count = int(targets.sum().item())
    return EvidenceCalibration(
        split_name=split_name,
        dataset_version=dataset_version,
        threshold=selected.threshold,
        precision=selected.precision,
        recall=selected.recall,
        f1=f1,
        average_precision=binary_average_precision(values, targets),
        positive_count=positive_count,
        negative_count=int(targets.numel()) - positive_count,
        curve=curve,
    )


def validate_calibration_source(
    *,
    split_name: str,
    dataset_version: str,
) -> None:
    """Reject sealed or historical data before a caller materialises it."""

    normalized = split_name.lower()
    if "sealed" in normalized or normalized.startswith("test"):
        raise ValueError("sealed/test data may not be used for calibration")
    allowed_versions = {
        "spider-programs-v0.2",
        "spider-programs-v0.3-recurrence-dev",
        "spider-programs-v0.4-aligned-dev",
    }
    if dataset_version not in allowed_versions:
        raise ValueError(
            "calibration requires a registered non-sealed Spider "
            "development dataset (v0.2, recurrence-dev, or v0.4)"
        )
