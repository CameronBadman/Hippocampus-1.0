from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _harmonic(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("metric components must be finite")
    if any(value <= 0 for value in values):
        return 0.0
    return len(values) / sum(1 / value for value in values)


def _average_precision(labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 1.0
    ranks = np.flatnonzero(labels) + 1
    return float(np.mean(np.arange(1, positives + 1) / ranks))


def evaluate_retrieval(
    *,
    scores: np.ndarray,
    relevance: np.ndarray,
    adversary: np.ndarray,
    tie_break: np.ndarray,
    null_scores: np.ndarray | None = None,
    scenario_families: Sequence[str] | None = None,
    recall_k: int = 8,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=bool)
    adversary = np.asarray(adversary, dtype=np.int16)
    tie_break = np.asarray(tie_break, dtype=np.int64)
    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must be aligned matrices")
    if adversary.shape != scores.shape or tie_break.shape != scores.shape:
        raise ValueError("adversary and tie-break matrices must align")
    if not np.isfinite(scores).all():
        raise ValueError("retrieval scores must be finite")
    case_count, candidate_count = scores.shape
    if not 0 < recall_k <= candidate_count:
        raise ValueError("recall_k must fit the candidate pool")
    orders = np.asarray(
        [np.lexsort((tie_break[index], -scores[index])) for index in range(case_count)]
    )
    reciprocal: list[float] = []
    recalls: list[float] = []
    average_precisions: list[float] = []
    exact_top_k = []
    for index, order in enumerate(orders):
        ranked = relevance[index, order]
        positive_count = int(ranked.sum())
        if positive_count:
            first = int(np.flatnonzero(ranked)[0])
            reciprocal.append(1 / (first + 1))
            recalls.append(float(ranked[:recall_k].sum() / positive_count))
            average_precisions.append(_average_precision(ranked))
            exact_top_k.append(bool(ranked[:positive_count].all()))
    if not reciprocal:
        raise ValueError("evaluation requires answerable cases")
    family_pairwise = []
    family_ids = sorted(set(int(value) for value in adversary.flat if value >= 0))
    for family_id in family_ids:
        correct = 0.0
        comparisons = 0
        for index in range(case_count):
            positives = scores[index, relevance[index]]
            negatives = scores[index, adversary[index] == family_id]
            if not positives.size or not negatives.size:
                continue
            differences = positives[:, None] - negatives[None, :]
            correct += float((differences > 0).sum())
            correct += 0.5 * float((differences == 0).sum())
            comparisons += int(differences.size)
        if comparisons:
            family_pairwise.append(correct / comparisons)
    macro_pairwise = float(np.mean(family_pairwise)) if family_pairwise else 0.0
    components = {
        "mrr": float(np.mean(reciprocal)),
        "recall_at_8": float(np.mean(recalls)),
        "macro_average_precision": float(np.mean(average_precisions)),
        "macro_hard_negative_pairwise_accuracy": macro_pairwise,
    }
    result: dict[str, Any] = {
        "score": _harmonic(tuple(components.values())),
        "components": components,
        "answerable_case_count": len(reciprocal),
        "case_count": case_count,
        "oracle_cardinality_top_rank_exact": float(np.mean(exact_top_k)),
    }
    if null_scores is not None:
        null = np.asarray(null_scores, dtype=np.float64)
        if null.shape != (case_count,) or not np.isfinite(null).all():
            raise ValueError("null_scores must be a finite case vector")
        selected = scores > null[:, None]
        true_positive = int((selected & relevance).sum())
        false_positive = int((selected & ~relevance).sum())
        false_negative = int((~selected & relevance).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        result["set_selection"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "exact_set_accuracy": float(np.all(selected == relevance, axis=1).mean()),
            "false_positives_per_case": false_positive / case_count,
            "mean_predicted_cardinality": float(selected.sum(axis=1).mean()),
            "mean_required_cardinality": float(relevance.sum(axis=1).mean()),
        }
    if scenario_families is not None:
        if len(scenario_families) != case_count:
            raise ValueError("scenario families must align with cases")
        per_family = {}
        for family in sorted(set(scenario_families)):
            indexes = [index for index, value in enumerate(scenario_families) if value == family]
            answerable = [index for index in indexes if relevance[index].any()]
            per_family[family] = {
                "case_count": len(indexes),
                "answerable_case_count": len(answerable),
                "mrr": float(
                    np.mean(
                        [
                            1 / (int(np.flatnonzero(relevance[index, orders[index]])[0]) + 1)
                            for index in answerable
                        ]
                    )
                ) if answerable else 0.0,
                "recall_at_8": float(
                    np.mean(
                        [
                            relevance[index, orders[index]][:recall_k].sum()
                            / relevance[index].sum()
                            for index in answerable
                        ]
                    )
                ) if answerable else 0.0,
            }
        result["per_scenario_family"] = per_family
    return result
