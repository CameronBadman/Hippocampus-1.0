from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from .schema import GraphProgramCase


@dataclass(frozen=True, slots=True)
class MetadataLeakageReport:
    case_count: int
    majority_accuracy: float
    best_stump_accuracy: float
    answerability_advantage: float
    fixed_answer_position_rate: float
    fixed_edge_position_rate: float


def _features(case: GraphProgramCase) -> tuple[float, ...]:
    start = case.start_nodes[0]
    out_degree = sum(
        edge.source_node == start
        or edge.bidirectional and edge.destination_node == start
        for edge in case.edges
    )
    summary_rows = [len(node.summary_atoms) for node in case.nodes]
    context_rows = [len(node.context_atoms) for node in case.nodes]
    return (
        float(len(case.nodes)),
        float(len(case.edges)),
        float(out_degree),
        float(sum(summary_rows)),
        float(sum(context_rows)),
        float(max(summary_rows)),
        float(max(context_rows, default=0)),
        float(case.search_budget),
        float(case.context_budget),
    )


def _majority(labels: Sequence[int]) -> int:
    return int(sum(labels) * 2 >= len(labels))


def _fit_stump(
    features: Sequence[tuple[float, ...]],
    labels: Sequence[int],
) -> tuple[int, float, bool]:
    best = (0, 0.0, True)
    best_accuracy = -1.0
    for feature_id in range(len(features[0])):
        values = sorted({row[feature_id] for row in features})
        thresholds = values or [0.0]
        for threshold in thresholds:
            for positive_above in (False, True):
                predictions = [
                    int((row[feature_id] >= threshold) == positive_above)
                    for row in features
                ]
                accuracy = mean(
                    prediction == label
                    for prediction, label in zip(
                        predictions,
                        labels,
                        strict=True,
                    )
                )
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best = (feature_id, threshold, positive_above)
    return best


def _mode_rate(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts = {value: values.count(value) for value in set(values)}
    return max(counts.values()) / len(values)


def metadata_leakage_report(
    cases: Sequence[GraphProgramCase],
    *,
    folds: int = 4,
) -> MetadataLeakageReport:
    if len(cases) < folds * 2:
        raise ValueError("leakage diagnostic requires at least two cases per fold")
    labels = [int(case.answerable) for case in cases]
    features = [_features(case) for case in cases]
    majority_predictions: list[int] = []
    stump_predictions: list[int] = []
    for fold in range(folds):
        test_ids = [index for index in range(len(cases)) if index % folds == fold]
        train_ids = [index for index in range(len(cases)) if index % folds != fold]
        train_labels = [labels[index] for index in train_ids]
        majority = _majority(train_labels)
        feature_id, threshold, positive_above = _fit_stump(
            [features[index] for index in train_ids],
            train_labels,
        )
        for index in test_ids:
            majority_predictions.append(majority)
            stump_predictions.append(
                int(
                    (features[index][feature_id] >= threshold)
                    == positive_above
                )
            )
    majority_accuracy = mean(
        prediction == label
        for prediction, label in zip(
            majority_predictions,
            labels,
            strict=True,
        )
    )
    stump_accuracy = mean(
        prediction == label
        for prediction, label in zip(
            stump_predictions,
            labels,
            strict=True,
        )
    )
    answer_positions = [
        case.answer_nodes[0] for case in cases if case.answer_nodes
    ]
    edge_positions = [
        candidate.edge_id
        for case in cases
        if case.answerable
        for round_ in case.trace.rounds
        for candidate in round_.candidates
        if candidate.acceptable
    ]
    return MetadataLeakageReport(
        case_count=len(cases),
        majority_accuracy=majority_accuracy,
        best_stump_accuracy=stump_accuracy,
        answerability_advantage=max(0.0, stump_accuracy - majority_accuracy),
        fixed_answer_position_rate=_mode_rate(answer_positions),
        fixed_edge_position_rate=_mode_rate(edge_positions),
    )
