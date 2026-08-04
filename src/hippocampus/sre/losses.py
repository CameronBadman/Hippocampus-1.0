from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import SRERetrievalOutput
from .packed import PackedSRERetrievalBatch


@dataclass(frozen=True, slots=True)
class SRERetrievalLossConfig:
    bce_weight: float = 1.0
    listwise_weight: float = 0.5
    hard_negative_weight: float = 0.25
    alignment_weight: float = 0.0
    ranking_margin: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "bce_weight",
            "listwise_weight",
            "hard_negative_weight",
            "alignment_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} may not be negative")


def _balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels_float = labels.to(logits.dtype)
    positive_count = labels_float.sum()
    negative_count = labels_float.numel() - positive_count
    positive_weight = negative_count / positive_count.clamp_min(1)
    weights = torch.where(
        labels,
        positive_weight.expand_as(labels_float),
        torch.ones_like(labels_float),
    )
    return F.binary_cross_entropy_with_logits(logits, labels_float, weight=weights)


def _multi_positive_mass(
    logits: torch.Tensor,
    positives: torch.Tensor,
) -> torch.Tensor:
    answerable = positives.any(dim=1)
    if not bool(answerable.any()):
        return logits.sum() * 0
    selected_logits = logits[answerable]
    selected_positive = positives[answerable]
    positive_values = selected_logits.masked_fill(~selected_positive, -torch.inf)
    return -(
        torch.logsumexp(positive_values, dim=1)
        - torch.logsumexp(selected_logits, dim=1)
    ).mean()


def _hard_negative_ranking(
    logits: torch.Tensor,
    positives: torch.Tensor,
    hard_negatives: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    losses = []
    for case_scores, case_positive, case_negative in zip(
        logits,
        positives,
        hard_negatives,
        strict=True,
    ):
        positive_scores = case_scores[case_positive]
        negative_scores = case_scores[case_negative]
        if positive_scores.numel() and negative_scores.numel():
            hardest = negative_scores.topk(min(8, negative_scores.numel())).values
            losses.append(
                F.softplus(
                    hardest[:, None] - positive_scores[None, :] + margin
                ).mean()
            )
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def sre_retrieval_loss(
    output: SRERetrievalOutput,
    batch: PackedSRERetrievalBatch,
    config: SRERetrievalLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    relative_scores = output.scores - output.null_scores[:, None]
    bce = _balanced_bce(relative_scores, batch.relevance)
    listwise = _multi_positive_mass(output.scores, batch.relevance)
    ranking = _hard_negative_ranking(
        output.scores,
        batch.relevance,
        batch.hard_negative,
        config.ranking_margin,
    )
    alignment = _multi_positive_mass(output.canonical_scores, batch.relevance)
    weighted = {
        "bce": config.bce_weight * bce,
        "listwise": config.listwise_weight * listwise,
        "hard_negative": config.hard_negative_weight * ranking,
        "alignment": config.alignment_weight * alignment,
    }
    total = sum(weighted.values(), start=output.scores.sum() * 0)
    components = {
        "total": total,
        "raw_bce": bce,
        "raw_listwise": listwise,
        "raw_hard_negative": ranking,
        "raw_alignment": alignment,
        **{f"weighted_{name}": value for name, value in weighted.items()},
    }
    return total, components
