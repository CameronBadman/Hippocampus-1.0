from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .config import ExecutionPolicy


INT32_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1


def require_int32_capacity(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if value > INT32_MAX:
        raise OverflowError(
            f"{name}={value} exceeds persistent int32 capacity ({INT32_MAX})"
        )
    return value


def checked_add(name: str, left: int, right: int, *, limit: int = INT64_MAX) -> int:
    left = int(left)
    right = int(right)
    if left < 0 or right < 0:
        raise ValueError(f"{name} operands must be non-negative")
    if left > limit - right:
        raise OverflowError(f"{name} addition exceeds capacity {limit}")
    return left + right


def checked_product(
    name: str, factors: Iterable[int], *, limit: int = min(INT64_MAX, sys.maxsize)
) -> int:
    result = 1
    for raw_factor in factors:
        factor = int(raw_factor)
        if factor < 0:
            raise ValueError(f"{name} factors must be non-negative")
        if factor and result > limit // factor:
            raise OverflowError(f"{name} product exceeds capacity {limit}")
        result *= factor
    return result


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def validate_tensor_allocation(
    name: str,
    shape: Iterable[int],
    dtype: torch.dtype,
    *,
    additional_element_factor: int = 1,
) -> tuple[int, int]:
    elements = checked_product(
        f"{name} element count", (*tuple(int(v) for v in shape), additional_element_factor)
    )
    byte_count = checked_product(
        f"{name} byte count", (elements, dtype_nbytes(dtype))
    )
    return elements, byte_count


def require_tensor(
    tensor: Any,
    name: str,
    *,
    ndim: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    contiguous: bool = False,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}, got shape {tuple(tensor.shape)}")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def require_floating(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype, got {tensor.dtype}")


def tensor_scalar_bool(value: torch.Tensor) -> bool:
    return bool(value.item())


def validate_pointer(
    pointer: torch.Tensor,
    name: str,
    *,
    expected_final: int | None = None,
) -> None:
    require_tensor(pointer, name, ndim=1, dtype=torch.int32, contiguous=True)
    if pointer.numel() == 0:
        raise ValueError(f"{name} must contain at least its leading zero")
    if int(pointer[0].item()) != 0:
        raise ValueError(f"{name}[0] must be zero")
    if pointer.numel() > 1 and tensor_scalar_bool(torch.any(pointer[1:] < pointer[:-1])):
        raise ValueError(f"{name} must be non-decreasing")
    final = int(pointer[-1].item())
    if final < 0:
        raise ValueError(f"{name} may not contain negative offsets")
    if expected_final is not None and final != int(expected_final):
        raise ValueError(
            f"{name}[-1] must equal {expected_final}, got {final}"
        )


def as_id_tensor(
    ids: Any,
    *,
    name: str,
    device: torch.device,
    validate_ids: bool,
    upper_bound: int,
) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        if ids.device != device:
            raise ValueError(f"{name} must be on {device}, got {ids.device}")
        if ids.ndim != 1:
            raise ValueError(f"{name} must be rank 1, got shape {tuple(ids.shape)}")
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must use int32 or int64, got {ids.dtype}")
        result = ids
    else:
        result = torch.as_tensor(ids, dtype=torch.int64, device=device)
        if result.ndim != 1:
            raise ValueError(f"{name} must be rank 1, got shape {tuple(result.shape)}")

    if validate_ids and result.numel():
        if tensor_scalar_bool(torch.any(result < 0)) or tensor_scalar_bool(
            torch.any(result >= upper_bound)
        ):
            low = int(result.min().item())
            high = int(result.max().item())
            raise IndexError(
                f"{name} contains IDs outside [0, {upper_bound}): min={low}, max={high}"
            )
    return result


def infer_common_device(
    tensors: Iterable[torch.Tensor], *, what: str
) -> torch.device | None:
    devices = {tensor.device for tensor in tensors}
    if not devices:
        return None
    if len(devices) != 1:
        rendered = ", ".join(sorted(str(device) for device in devices))
        raise ValueError(
            f"default packing requires one common source device for {what}; got {rendered}"
        )
    return next(iter(devices))


def infer_common_dtype(
    tensors: Iterable[torch.Tensor], *, what: str
) -> torch.dtype | None:
    dtypes = {tensor.dtype for tensor in tensors}
    if not dtypes:
        return None
    if len(dtypes) != 1:
        rendered = ", ".join(sorted(str(dtype) for dtype in dtypes))
        raise ValueError(
            f"default packing requires one common source dtype for {what}; got {rendered}"
        )
    return next(iter(dtypes))


def resolve_accumulation_dtype(
    dtype: torch.dtype, requested: torch.dtype | None
) -> torch.dtype:
    if requested is not None:
        if not torch.empty((), dtype=requested).is_floating_point():
            raise TypeError("accumulation_dtype must be floating point")
        return requested
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def next_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError("next_power_of_two requires a positive value")
    return 1 << (value - 1).bit_length()


def fill_like_shape(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    fill_value: float | int | torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(fill_value, torch.Tensor):
        fill = fill_value.to(device=reference.device, dtype=dtype)
        return torch.broadcast_to(fill, shape).clone()
    if isinstance(fill_value, float) and not math.isfinite(fill_value):
        return torch.full(shape, fill_value, device=reference.device, dtype=dtype)
    return torch.full(shape, fill_value, device=reference.device, dtype=dtype)


def policy_mode(policy: "ExecutionPolicy | str | None") -> str:
    from .config import resolve_execution_policy

    return resolve_execution_policy(policy).mode

