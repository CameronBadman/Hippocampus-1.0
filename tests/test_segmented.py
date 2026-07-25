from __future__ import annotations

import pytest
import torch

from hippocampus import (
    ExecutionPolicy,
    SegmentReduction,
    SegmentTransform,
    segment_broadcast,
    segment_logsumexp,
    segment_max,
    segment_mean,
    segment_softmax,
    segment_sum,
    segment_weighted_mean,
    segment_weighted_sum,
)


@pytest.fixture
def segmented_values():
    values = torch.tensor(
        [
            [1.0, 4.0],
            [3.0, 2.0],
            [2.0, 8.0],
            [6.0, 4.0],
            [5.0, 1.0],
        ],
        requires_grad=True,
    )
    offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    return values, offsets


def test_sum_mean_max_contracts(segmented_values) -> None:
    values, offsets = segmented_values
    summed = segment_sum(values, offsets)
    mean = segment_mean(values, offsets)
    maximum = segment_max(values, offsets)
    assert isinstance(summed, SegmentReduction)
    assert summed.valid_mask.tolist() == [True, False, True]
    assert summed.values.tolist() == [[4, 6], [0, 0], [13, 13]]
    assert torch.allclose(
        mean.values,
        torch.tensor([[2, 3], [0, 0], [13 / 3, 13 / 3]]),
    )
    assert maximum.values.tolist() == [[3, 4], [0, 0], [6, 8]]


def test_configurable_empty_fill(segmented_values) -> None:
    values, offsets = segmented_values
    result = segment_mean(values, offsets, fill_value=-7)
    assert result.values[1].tolist() == [-7, -7]
    assert not result.valid_mask[1]


def test_stable_logsumexp_and_gradient() -> None:
    values = torch.tensor(
        [[10_000.0], [10_001.0], [-10_000.0]],
        requires_grad=True,
    )
    result = segment_logsumexp(values, [0, 2, 2, 3])
    expected = torch.tensor(
        [[10_001.3132617], [0.0], [-10_000.0]]
    )
    assert torch.allclose(result.values, expected)
    assert torch.isfinite(result.values).all()
    result.values.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_weighted_reductions_mark_zero_total_weight_invalid(
    segmented_values,
) -> None:
    values, offsets = segmented_values
    weights = torch.tensor([1.0, -1.0, 3.0, 1.0, 0.0], requires_grad=True)
    summed = segment_weighted_sum(values, weights, offsets)
    mean = segment_weighted_mean(values, weights, offsets)
    assert summed.valid_mask.tolist() == [False, False, True]
    assert summed.values[0].tolist() == [0, 0]
    assert mean.values[2].tolist() == [3, 7]
    mean.values.sum().backward()
    assert values.grad is not None
    assert weights.grad is not None


def test_segment_softmax_is_stable_and_row_preserving(
    segmented_values,
) -> None:
    values, offsets = segmented_values
    transformed = segment_softmax(values * 1_000, offsets)
    assert isinstance(transformed, SegmentTransform)
    assert transformed.values.shape == values.shape
    assert transformed.valid_mask.tolist() == [True, False, True]
    assert torch.allclose(
        transformed.values[:2].sum(dim=0), torch.ones(2)
    )
    assert torch.allclose(
        transformed.values[2:].sum(dim=0), torch.ones(2)
    )
    transformed.values.sum().backward()
    assert values.grad is not None


def test_segment_broadcast_retains_empty_validity() -> None:
    segment_values = torch.tensor([[1.0], [2.0], [3.0]])
    result = segment_broadcast(segment_values, [0, 2, 2, 5])
    assert result.values.flatten().tolist() == [1, 1, 3, 3, 3]
    assert result.valid_mask.tolist() == [True, False, True]


def test_row_owner_ids_accept_noncontiguous_occurrences() -> None:
    values = torch.tensor([[1.0], [10.0], [2.0], [20.0]])
    owners = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    result = segment_sum(
        values, row_owner_ids=owners, num_segments=3
    )
    assert result.values.flatten().tolist() == [3, 30, 0]
    assert result.valid_mask.tolist() == [True, True, False]
    softmax = segment_softmax(
        values, row_owner_ids=owners, num_segments=3
    )
    assert torch.allclose(
        softmax.values[[0, 2]].sum(), torch.tensor(1.0)
    )


def test_half_reductions_accumulate_in_fp32_and_return_input_dtype() -> None:
    values = torch.full((2_000, 1), 0.1, dtype=torch.float16)
    result = segment_sum(values, [0, 2_000])
    assert result.values.dtype == torch.float16
    assert torch.allclose(
        result.values.float(), torch.tensor([[200.0]]), atol=0.2
    )
    fp32 = segment_sum(values, [0, 2_000], output_dtype=torch.float32)
    assert fp32.values.dtype == torch.float32
    assert torch.allclose(fp32.values, torch.tensor([[200.0]]), atol=0.1)


def test_empty_values_and_empty_segments_have_canonical_shapes() -> None:
    values = torch.empty(0, 3, requires_grad=True)
    result = segment_logsumexp(values, [0, 0, 0])
    transform = segment_softmax(values, [0, 0, 0])
    assert result.values.shape == (2, 3)
    assert result.valid_mask.tolist() == [False, False]
    assert transform.values.shape == (0, 3)
    assert transform.valid_mask.tolist() == [False, False]


@pytest.mark.parametrize(
    "operation",
    [
        lambda values, offsets: segment_sum(values, offsets),
        lambda values, offsets: segment_mean(values, offsets),
        lambda values, offsets: segment_logsumexp(values, offsets),
        lambda values, offsets: segment_softmax(values, offsets),
    ],
)
def test_segment_operations_pass_gradcheck(operation) -> None:
    values = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
    assert torch.autograd.gradcheck(
        lambda input_values: operation(input_values, offsets).values,
        (values,),
    )


def test_weighted_mean_passes_gradcheck() -> None:
    values = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.rand(5, dtype=torch.float64, requires_grad=True) + 0.2
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
    assert torch.autograd.gradcheck(
        lambda input_values, input_weights: segment_weighted_mean(
            input_values, input_weights, offsets
        ).values,
        (values, weights),
    )


def test_deterministic_policy_requires_global_state() -> None:
    torch.use_deterministic_algorithms(False)
    with pytest.raises(RuntimeError, match="use_deterministic_algorithms"):
        ExecutionPolicy("deterministic")
    with pytest.raises(RuntimeError, match="use_deterministic_algorithms"):
        segment_sum(
            torch.ones(2, 1),
            [0, 2],
            execution_policy="deterministic",
        )


def test_deterministic_operations_replay() -> None:
    values = torch.randn(20, 3)
    offsets = torch.tensor([0, 3, 3, 11, 20], dtype=torch.int32)
    torch.use_deterministic_algorithms(True)
    try:
        first = segment_logsumexp(
            values, offsets, execution_policy="deterministic"
        )
        second = segment_logsumexp(
            values, offsets, execution_policy="deterministic"
        )
        assert torch.equal(first.values, second.values)
        first_softmax = segment_softmax(
            values, offsets, execution_policy="deterministic"
        )
        second_softmax = segment_softmax(
            values, offsets, execution_policy="deterministic"
        )
        assert torch.equal(first_softmax.values, second_softmax.values)
    finally:
        torch.use_deterministic_algorithms(False)


def test_invalid_segment_inputs_are_rejected() -> None:
    values = torch.ones(2, 1)
    with pytest.raises(ValueError, match="exactly one"):
        segment_sum(values)
    with pytest.raises(ValueError, match="non-decreasing"):
        segment_sum(values, [0, 2, 1])
    with pytest.raises(IndexError, match="out-of-range"):
        segment_sum(
            values,
            row_owner_ids=torch.tensor([0, 2]),
            num_segments=2,
        )
