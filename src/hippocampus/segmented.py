from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ._validation import (
    fill_like_shape,
    require_floating,
    require_tensor,
    resolve_accumulation_dtype,
    tensor_scalar_bool,
    validate_tensor_allocation,
)
from .config import ExecutionMode, ExecutionPolicy, resolve_execution_policy


ReductionKind = Literal[
    "sum", "mean", "max", "logsumexp", "weighted_sum", "weighted_mean"
]


@dataclass(frozen=True, slots=True)
class SegmentReduction:
    values: torch.Tensor
    valid_mask: torch.Tensor

    @property
    def result(self) -> torch.Tensor:
        return self.values

    def __iter__(self):
        yield self.values
        yield self.valid_mask


@dataclass(frozen=True, slots=True)
class SegmentTransform:
    values: torch.Tensor
    valid_mask: torch.Tensor

    @property
    def result(self) -> torch.Tensor:
        return self.values

    def __iter__(self):
        yield self.values
        yield self.valid_mask


@dataclass(frozen=True, slots=True)
class _Segments:
    row_owner_ids: torch.Tensor
    segment_count: int
    row_count: int
    counts: torch.Tensor


def _as_integer_vector(
    value: Any, *, name: str, device: torch.device
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        require_tensor(value, name, ndim=1, device=device)
        if value.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"{name} must use an integer dtype")
        return value.to(torch.int64)
    result = torch.as_tensor(value, dtype=torch.int64, device=device)
    if result.ndim != 1:
        raise ValueError(f"{name} must be rank 1")
    return result


def _resolve_segments(
    *,
    row_count: int,
    device: torch.device,
    offsets: Any | None,
    row_owner_ids: Any | None,
    num_segments: int | None,
) -> _Segments:
    if (offsets is None) == (row_owner_ids is None):
        raise ValueError("provide exactly one of offsets or row_owner_ids")

    if offsets is not None:
        pointer = _as_integer_vector(
            offsets, name="offsets", device=device
        )
        if pointer.numel() == 0:
            raise ValueError("offsets must contain at least a leading zero")
        if int(pointer[0].item()) != 0:
            raise ValueError("offsets[0] must be zero")
        if pointer.numel() > 1 and tensor_scalar_bool(
            torch.any(pointer[1:] < pointer[:-1])
        ):
            raise ValueError("offsets must be non-decreasing")
        if int(pointer[-1].item()) != row_count:
            raise ValueError(
                f"offsets[-1] must equal values row count {row_count}"
            )
        inferred_count = int(pointer.numel() - 1)
        if num_segments is not None and num_segments != inferred_count:
            raise ValueError("num_segments disagrees with offsets")
        segment_count = inferred_count
        counts = pointer[1:] - pointer[:-1]
        owners = torch.repeat_interleave(
            torch.arange(segment_count, dtype=torch.int64, device=device),
            counts,
        )
    else:
        owners = _as_integer_vector(
            row_owner_ids, name="row_owner_ids", device=device
        )
        if owners.numel() != row_count:
            raise ValueError("row_owner_ids must align with values rows")
        if num_segments is None:
            segment_count = int(owners.max().item()) + 1 if row_count else 0
        else:
            if isinstance(num_segments, bool) or not isinstance(num_segments, int):
                raise TypeError("num_segments must be an integer")
            if num_segments < 0:
                raise ValueError("num_segments must be non-negative")
            segment_count = num_segments
        if row_count and (
            tensor_scalar_bool(torch.any(owners < 0))
            or tensor_scalar_bool(torch.any(owners >= segment_count))
        ):
            raise IndexError("row_owner_ids contains an out-of-range segment ID")
        counts = torch.bincount(owners, minlength=segment_count)

    validate_tensor_allocation(
        "segment validity mask", (segment_count,), torch.bool
    )
    return _Segments(
        row_owner_ids=owners.contiguous(),
        segment_count=segment_count,
        row_count=row_count,
        counts=counts.to(torch.int64).contiguous(),
    )


def _flatten_values(
    values: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    require_tensor(values, "values")
    require_floating(values, "values")
    if values.ndim < 1:
        raise ValueError("values must have a row dimension")
    feature_shape = tuple(int(size) for size in values.shape[1:])
    flattened_width = 1
    for size in feature_shape:
        flattened_width *= size
    return values.reshape(values.shape[0], flattened_width), feature_shape


def _restore_feature_shape(
    flattened: torch.Tensor,
    *,
    leading_rows: int,
    feature_shape: tuple[int, ...],
) -> torch.Tensor:
    return flattened.reshape((leading_rows, *feature_shape))


def _flat_fill(
    reference: torch.Tensor,
    feature_shape: tuple[int, ...],
    fill_value: float | int | torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    shape = feature_shape or (1,)
    return fill_like_shape(
        reference, shape, fill_value, dtype=dtype
    ).reshape(1, -1)


def _output_dtype(
    values: torch.Tensor, requested: torch.dtype | None
) -> torch.dtype:
    dtype = requested or values.dtype
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("output_dtype must be floating point")
    return dtype


def _prepare_reduction(
    values: torch.Tensor,
    *,
    offsets: Any | None,
    row_owner_ids: Any | None,
    num_segments: int | None,
    accumulation_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
    execution_policy: ExecutionPolicy | ExecutionMode | None,
) -> tuple[
    torch.Tensor,
    tuple[int, ...],
    _Segments,
    torch.dtype,
    torch.dtype,
    ExecutionPolicy,
]:
    flat, feature_shape = _flatten_values(values)
    segments = _resolve_segments(
        row_count=int(values.shape[0]),
        device=values.device,
        offsets=offsets,
        row_owner_ids=row_owner_ids,
        num_segments=num_segments,
    )
    accumulation = resolve_accumulation_dtype(
        values.dtype, accumulation_dtype
    )
    output = _output_dtype(values, output_dtype)
    policy = resolve_execution_policy(execution_policy)
    validate_tensor_allocation(
        "segment reduction output",
        (segments.segment_count, *feature_shape),
        output,
    )
    return (
        flat.to(accumulation),
        feature_shape,
        segments,
        accumulation,
        output,
        policy,
    )


def _fast_segment_sum(
    flat: torch.Tensor, segments: _Segments
) -> torch.Tensor:
    result = flat.new_zeros((segments.segment_count, flat.shape[1]))
    if segments.row_count:
        result = result.index_add(0, segments.row_owner_ids, flat)
    return result


def _fast_segment_max(
    flat: torch.Tensor, segments: _Segments
) -> torch.Tensor:
    result = torch.full(
        (segments.segment_count, flat.shape[1]),
        -torch.inf,
        dtype=flat.dtype,
        device=flat.device,
    )
    if segments.row_count:
        indices = segments.row_owner_ids[:, None].expand_as(flat)
        result = result.scatter_reduce(
            0, indices, flat, reduce="amax", include_self=True
        )
    return result


def _apply_invalid_fill(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    fill: torch.Tensor,
) -> torch.Tensor:
    if values.shape[0] == 0:
        return values
    return torch.where(valid[:, None], values, fill.expand_as(values))


def _deterministic_reduce(
    flat: torch.Tensor,
    segments: _Segments,
    *,
    kind: ReductionKind,
    weights: torch.Tensor | None,
    fill: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_by_segment = segments.row_owner_ids
    outputs: list[torch.Tensor] = []
    validity: list[bool] = []
    for segment_id in range(segments.segment_count):
        row_indices = torch.nonzero(
            rows_by_segment == segment_id
        ).flatten()
        segment_values = flat.index_select(0, row_indices)
        if row_indices.numel() == 0:
            outputs.append(fill.squeeze(0))
            validity.append(False)
            continue

        if kind == "sum":
            result = segment_values.sum(dim=0)
            valid = True
        elif kind == "mean":
            result = segment_values.mean(dim=0)
            valid = True
        elif kind == "max":
            result = segment_values.amax(dim=0)
            valid = True
        elif kind == "logsumexp":
            maximum = segment_values.amax(dim=0)
            result = maximum + torch.log(
                torch.exp(segment_values - maximum).sum(dim=0)
            )
            valid = True
        else:
            assert weights is not None
            segment_weights = weights.index_select(0, row_indices)
            weight_total = segment_weights.sum()
            valid = bool((weight_total != 0).item())
            if valid:
                weighted = (
                    segment_values * segment_weights[:, None]
                ).sum(dim=0)
                result = (
                    weighted / weight_total
                    if kind == "weighted_mean"
                    else weighted
                )
            else:
                result = fill.squeeze(0)
        outputs.append(result)
        validity.append(valid)

    if outputs:
        result_values = torch.stack(outputs, dim=0)
    else:
        result_values = flat.new_empty((0, flat.shape[1]))
    valid_mask = torch.tensor(
        validity, dtype=torch.bool, device=flat.device
    )
    return result_values, valid_mask


def _reduce(
    values: torch.Tensor,
    kind: ReductionKind,
    offsets: Any | None = None,
    *,
    row_owner_ids: Any | None = None,
    num_segments: int | None = None,
    weights: torch.Tensor | None = None,
    fill_value: float | int | torch.Tensor = 0,
    accumulation_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
) -> SegmentReduction:
    (
        flat,
        feature_shape,
        segments,
        accumulation,
        output,
        policy,
    ) = _prepare_reduction(
        values,
        offsets=offsets,
        row_owner_ids=row_owner_ids,
        num_segments=num_segments,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        execution_policy=execution_policy,
    )
    fill = _flat_fill(values, feature_shape, fill_value, accumulation)

    weight_values: torch.Tensor | None = None
    if kind in ("weighted_sum", "weighted_mean"):
        if weights is None:
            raise ValueError(f"{kind} requires weights")
        require_tensor(weights, "weights", ndim=1, device=values.device)
        require_floating(weights, "weights")
        if weights.numel() != segments.row_count:
            raise ValueError("weights must align with values rows")
        weight_values = weights.to(accumulation)
    elif weights is not None:
        raise ValueError(f"{kind} does not accept weights")

    if policy.mode == "deterministic":
        reduced, valid = _deterministic_reduce(
            flat,
            segments,
            kind=kind,
            weights=weight_values,
            fill=fill,
        )
    else:
        valid = segments.counts > 0
        if kind == "sum":
            reduced = _fast_segment_sum(flat, segments)
        elif kind == "mean":
            reduced = _fast_segment_sum(flat, segments)
            denominator = segments.counts.clamp_min(1).to(flat.dtype)[:, None]
            reduced = reduced / denominator
        elif kind == "max":
            reduced = _fast_segment_max(flat, segments)
        elif kind == "logsumexp":
            maximum = _fast_segment_max(flat, segments)
            row_maximum = maximum[segments.row_owner_ids]
            exponentials = torch.exp(flat - row_maximum)
            exponential_sum = _fast_segment_sum(exponentials, segments)
            reduced = maximum + torch.log(exponential_sum.clamp_min(torch.finfo(flat.dtype).tiny))
        else:
            assert weight_values is not None
            weight_totals = weight_values.new_zeros(segments.segment_count)
            if segments.row_count:
                weight_totals = weight_totals.index_add(
                    0, segments.row_owner_ids, weight_values
                )
            valid = valid & (weight_totals != 0)
            reduced = _fast_segment_sum(
                flat * weight_values[:, None], segments
            )
            if kind == "weighted_mean":
                safe_totals = torch.where(
                    weight_totals != 0,
                    weight_totals,
                    torch.ones_like(weight_totals),
                )
                reduced = reduced / safe_totals[:, None]
        reduced = _apply_invalid_fill(reduced, valid, fill=fill)

    restored = _restore_feature_shape(
        reduced.to(output),
        leading_rows=segments.segment_count,
        feature_shape=feature_shape,
    )
    return SegmentReduction(restored, valid.contiguous())


def segment_sum(
    values: torch.Tensor,
    offsets: Any | None = None,
    *,
    row_owner_ids: Any | None = None,
    num_segments: int | None = None,
    fill_value: float | int | torch.Tensor = 0,
    accumulation_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
) -> SegmentReduction:
    return _reduce(
        values,
        "sum",
        offsets,
        row_owner_ids=row_owner_ids,
        num_segments=num_segments,
        fill_value=fill_value,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        execution_policy=execution_policy,
    )


def segment_mean(
    values: torch.Tensor,
    offsets: Any | None = None,
    *,
    row_owner_ids: Any | None = None,
    num_segments: int | None = None,
    fill_value: float | int | torch.Tensor = 0,
    accumulation_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
) -> SegmentReduction:
    return _reduce(
        values,
        "mean",
        offsets,
        row_owner_ids=row_owner_ids,
        num_segments=num_segments,
        fill_value=fill_value,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        execution_policy=execution_policy,
    )


def segment_max(
    values: torch.Tensor,
    offsets: Any | None = None,
    *,
    row_owner_ids: Any | None = None,
    num_segments: int | None = None,
    fill_value: float | int | torch.Tensor = 0,
    accumulation_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
) -> SegmentReduction:
    return _reduce(
        values,
        "max",
        offsets,
        row_owner_ids=row_owner_ids,
        num_segments=num_segments,
        fill_value=fill_value,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        execution_policy=execution_policy,
    )


def segment_logsumexp(
    values: torch.Tensor,
    offsets: Any | None = None,
    *,
    row_owner_ids: Any | None = None,
    num_segments: int | None = None,
    fill_value: float | int | torch.Tensor = 0,
    accumulation_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
) -> SegmentReduction:
    return _reduce(
        values,
        "logsumexp",
        offsets,
        row_owner_ids=row_owner_ids,
        num_segments=num_segments,
        fill_value=fill_value,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        execution_policy=execution_policy,
    )


def segment_weighted_sum(
    values: torch.Tensor,
    weights: torch.Tensor,
    offsets: Any | None = None,
    *,
    row_owner_ids: Any | None = None,
    num_segments: int | None = None,
    fill_value: float | int | torch.Tensor = 0,
    accumulation_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
) -> SegmentReduction:
    return _reduce(
        values,
        "weighted_sum",
        offsets,
        row_owner_ids=row_owner_ids,
        num_segments=num_segments,
        weights=weights,
        fill_value=fill_value,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        execution_policy=execution_policy,
    )


def segment_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    offsets: Any | None = None,
    *,
    row_owner_ids: Any | None = None,
    num_segments: int | None = None,
    fill_value: float | int | torch.Tensor = 0,
    accumulation_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
) -> SegmentReduction:
    return _reduce(
        values,
        "weighted_mean",
        offsets,
        row_owner_ids=row_owner_ids,
        num_segments=num_segments,
        weights=weights,
        fill_value=fill_value,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        execution_policy=execution_policy,
    )


def segment_softmax(
    values: torch.Tensor,
    offsets: Any | None = None,
    *,
    row_owner_ids: Any | None = None,
    num_segments: int | None = None,
    accumulation_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
) -> SegmentTransform:
    (
        flat,
        feature_shape,
        segments,
        _,
        output,
        policy,
    ) = _prepare_reduction(
        values,
        offsets=offsets,
        row_owner_ids=row_owner_ids,
        num_segments=num_segments,
        accumulation_dtype=accumulation_dtype,
        output_dtype=output_dtype,
        execution_policy=execution_policy,
    )
    valid = segments.counts > 0
    if policy.mode == "deterministic":
        transformed = torch.zeros_like(flat)
        for segment_id in range(segments.segment_count):
            rows = torch.nonzero(
                segments.row_owner_ids == segment_id
            ).flatten()
            if rows.numel() == 0:
                continue
            segment_values = flat.index_select(0, rows)
            maximum = segment_values.amax(dim=0)
            exponentials = torch.exp(segment_values - maximum)
            probabilities = exponentials / exponentials.sum(dim=0)
            transformed = transformed.index_copy(0, rows, probabilities)
    else:
        maximum = _fast_segment_max(flat, segments)
        exponentials = torch.exp(
            flat - maximum[segments.row_owner_ids]
        )
        denominator = _fast_segment_sum(exponentials, segments)
        transformed = (
            exponentials / denominator[segments.row_owner_ids]
            if segments.row_count
            else exponentials
        )
    restored = _restore_feature_shape(
        transformed.to(output),
        leading_rows=segments.row_count,
        feature_shape=feature_shape,
    )
    return SegmentTransform(restored, valid.contiguous())


def segment_broadcast(
    segment_values: torch.Tensor,
    offsets: Any | None = None,
    *,
    row_owner_ids: Any | None = None,
    num_segments: int | None = None,
    output_dtype: torch.dtype | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
) -> SegmentTransform:
    require_tensor(segment_values, "segment_values")
    require_floating(segment_values, "segment_values")
    if segment_values.ndim < 1:
        raise ValueError("segment_values must have a segment dimension")
    policy = resolve_execution_policy(execution_policy)
    del policy  # Advanced indexing is deterministic for this one-to-many mapping.

    if offsets is not None:
        pointer = _as_integer_vector(
            offsets, name="offsets", device=segment_values.device
        )
        if pointer.numel() == 0:
            raise ValueError("offsets must contain at least a leading zero")
        row_count = int(pointer[-1].item())
    elif row_owner_ids is not None:
        owners = _as_integer_vector(
            row_owner_ids,
            name="row_owner_ids",
            device=segment_values.device,
        )
        row_count = int(owners.numel())
    else:
        raise ValueError("provide exactly one of offsets or row_owner_ids")

    expected_segments = int(segment_values.shape[0])
    if num_segments is not None and num_segments != expected_segments:
        raise ValueError("num_segments must equal segment_values.shape[0]")
    segments = _resolve_segments(
        row_count=row_count,
        device=segment_values.device,
        offsets=offsets,
        row_owner_ids=row_owner_ids,
        num_segments=expected_segments,
    )
    dtype = _output_dtype(segment_values, output_dtype)
    transformed = segment_values[segments.row_owner_ids].to(dtype)
    return SegmentTransform(transformed, (segments.counts > 0).contiguous())


segment_log_sum_exp = segment_logsumexp
segmented_sum = segment_sum
segmented_mean = segment_mean
segmented_max = segment_max
segmented_logsumexp = segment_logsumexp
segmented_log_sum_exp = segment_logsumexp
segmented_weighted_sum = segment_weighted_sum
segmented_weighted_mean = segment_weighted_mean
segmented_softmax = segment_softmax
segmented_broadcast = segment_broadcast
