from __future__ import annotations

import pytest
import torch

from hippocampus.spider import (
    CandidateOutputs,
    CandidateSupervision,
    SpiderLossConfig,
    SpiderModel,
    SpiderModelConfig,
    binary_average_precision,
    calibrate_evidence_threshold,
    candidate_loss_report,
    termination_loss_report,
)


def _outputs(evidence_logits: torch.Tensor) -> CandidateOutputs:
    count = evidence_logits.numel()
    zero = evidence_logits.new_zeros((count,))
    return CandidateOutputs(
        next_path_state=evidence_logits.new_zeros((count, 2, 4)),
        priority_logits=zero,
        expand_logits=zero,
        context_logits=zero,
        evidence_logits=evidence_logits,
        remaining_cost=zero,
        support_logits=zero,
        conflict_logits=zero,
    )


def _supervision(labels: torch.Tensor) -> CandidateSupervision:
    count = labels.numel()
    return CandidateSupervision(
        acceptable=torch.zeros(count, dtype=torch.bool),
        context_has_value=torch.zeros(count, dtype=torch.bool),
        include_as_evidence=labels,
        remaining_cost=torch.zeros(count),
        support=torch.zeros(count),
        conflict=torch.zeros(count),
    )


def test_balanced_evidence_loss_penalises_missed_positive() -> None:
    logits = torch.tensor([-2.0, -2.0, -2.0, -2.0], requires_grad=True)
    labels = torch.tensor([True, False, False, False])
    report = candidate_loss_report(
        _outputs(logits),
        _supervision(labels),
        torch.zeros(4, dtype=torch.int64),
        frontier_count=1,
        config=SpiderLossConfig(
            evidence_mode="balanced",
            evidence_set=1.0,
        ),
    )

    assert report.terms["evidence"].raw > 0
    assert report.terms["evidence_set"].raw > 0
    report.total.backward()
    assert logits.grad is not None
    assert logits.grad[0] < 0
    assert abs(logits.grad[0]) > abs(logits.grad[1])


def test_evidence_metrics_report_average_precision_and_label_counts() -> None:
    scores = torch.tensor([3.0, 2.0, -1.0, -2.0])
    labels = torch.tensor([True, False, True, False])

    average_precision = binary_average_precision(scores, labels)

    assert average_precision == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert int(labels.sum()) == 2
    assert int((~labels).sum()) == 2


def test_threshold_calibration_rejects_sealed_and_v0_data() -> None:
    scores = torch.tensor([3.0, 1.0, -1.0, -3.0])
    labels = torch.tensor([True, True, False, False])

    result = calibrate_evidence_threshold(
        scores,
        labels,
        split_name="validation_id",
        dataset_version="spider-programs-v0.2",
    )

    assert 0.0 <= result.threshold <= 1.0
    assert result.f1 == pytest.approx(1.0)
    with pytest.raises(ValueError, match="sealed"):
        calibrate_evidence_threshold(
            scores,
            labels,
            split_name="test_sealed",
            dataset_version="spider-programs-v0.2",
        )
    with pytest.raises(ValueError, match="v0.2"):
        calibrate_evidence_threshold(
            scores,
            labels,
            split_name="validation_id",
            dataset_version="spider-programs-v0.1",
        )


def test_hierarchical_termination_has_masked_losses_and_gradients() -> None:
    model = SpiderModel(
        SpiderModelConfig(
            summary_dim=8,
            context_dim=8,
            edge_dim=8,
            query_dim=8,
            d_model=16,
            num_heads=4,
            num_blocks=1,
            path_rows=3,
            evidence_rows=3,
            termination_mode="hierarchical",
        )
    )
    output = model.termination_head(
        torch.randn(4, 16),
        torch.randn(4, 16),
        torch.randn(4, 16),
        torch.randn(4, 6),
    )
    targets = torch.tensor([0, 1, 2, 5])
    report = termination_loss_report(
        output,
        targets,
        config=SpiderLossConfig(),
    )

    assert output.logits.shape == (4, 6)
    assert report.terms["termination_stop"].target_count == 4
    assert report.terms["termination_answer"].target_count == 3
    assert report.terms["termination_unknown"].target_count == 2
    report.total.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.termination_head.parameters()
    )
