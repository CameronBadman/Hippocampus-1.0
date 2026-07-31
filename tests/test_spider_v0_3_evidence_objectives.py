from __future__ import annotations

import pytest
import torch

from hippocampus.spider import (
    CandidateOutputs,
    CandidateSupervision,
    SpiderLossConfig,
    candidate_loss_report,
    multi_positive_evidence_ranking_loss,
)


def _outputs(evidence_logits: torch.Tensor) -> CandidateOutputs:
    count = evidence_logits.numel()
    zeros = evidence_logits.new_zeros((count,))
    return CandidateOutputs(
        next_path_state=evidence_logits.new_zeros((count, 2, 4)),
        priority_logits=zeros,
        expand_logits=zeros,
        context_logits=zeros,
        evidence_logits=evidence_logits,
        remaining_cost=zeros,
        support_logits=zeros,
        conflict_logits=zeros,
    )


def _supervision(
    labels: torch.Tensor,
    plausible: torch.Tensor,
) -> CandidateSupervision:
    count = labels.numel()
    return CandidateSupervision(
        acceptable=torch.zeros(count, dtype=torch.bool),
        context_has_value=torch.zeros(count, dtype=torch.bool),
        include_as_evidence=labels,
        remaining_cost=torch.zeros(count),
        support=torch.zeros(count),
        conflict=torch.zeros(count),
        evidence_plausible_negative=plausible,
    )


def test_multi_positive_ranking_requires_every_positive_to_outrank() -> None:
    logits = torch.tensor([2.0, -1.0, 1.5, 0.5], requires_grad=True)
    positives = torch.tensor([True, True, False, False])
    negatives = ~positives

    loss, count = multi_positive_evidence_ranking_loss(
        logits,
        positives,
        negatives,
        margin=0.2,
        max_negatives=2,
    )
    loss.backward()

    assert count == 2
    assert logits.grad is not None
    assert logits.grad[0] < 0
    assert logits.grad[1] < 0
    assert logits.grad[2] > 0
    assert logits.grad[3] > 0
    assert abs(logits.grad[1]) > abs(logits.grad[0])


def test_ranking_uses_only_registered_number_of_hard_negatives() -> None:
    logits = torch.tensor([0.0, 3.0, 2.0, -10.0], requires_grad=True)
    positives = torch.tensor([True, False, False, False])

    loss, _ = multi_positive_evidence_ranking_loss(
        logits,
        positives,
        ~positives,
        margin=0.2,
        max_negatives=2,
    )
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[1] > 0
    assert logits.grad[2] > 0
    assert logits.grad[3] == pytest.approx(0.0)


def test_e2_adds_plausible_ranking_without_replacing_e1_term() -> None:
    logits = torch.tensor([0.0, 2.0, 1.0, -2.0], requires_grad=True)
    labels = torch.tensor([True, False, False, False])
    plausible = torch.tensor([False, False, True, False])
    report = candidate_loss_report(
        _outputs(logits),
        _supervision(labels, plausible),
        torch.zeros(4, dtype=torch.int64),
        frontier_count=1,
        config=SpiderLossConfig(
            evidence_mode="balanced",
            evidence_positive_weight=4.0,
            evidence_set=0.0,
            evidence_ranking=0.5,
            evidence_plausible_ranking=0.5,
            evidence_ranking_margin=0.2,
            evidence_hard_negative_count=4,
        ),
    )

    assert report.terms["evidence_ranking"].target_count == 1
    assert report.terms["evidence_plausible_ranking"].target_count == 1
    assert report.terms["evidence_ranking"].raw > 0
    assert report.terms["evidence_plausible_ranking"].raw > 0
    report.total.backward()
    assert logits.grad is not None
    assert logits.grad[0] < 0
