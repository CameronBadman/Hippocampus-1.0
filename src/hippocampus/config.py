from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


ExecutionMode = Literal["fast", "deterministic"]
PersistentIndexDtype = Literal["int32"]
EphemeralIndexDtype = Literal["auto", "int32", "int64"]


@dataclass(frozen=True, slots=True)
class GraphSchema:
    """Logical manifold widths and empty-manifold invariants for a graph."""

    summary_dim: int
    context_dim: int
    edge_dim: int
    allow_empty_edge_manifolds: bool = False

    def __post_init__(self) -> None:
        for name in ("summary_dim", "context_dim", "edge_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class IndexPolicy:
    persistent: PersistentIndexDtype = "int32"
    ephemeral: EphemeralIndexDtype = "auto"

    def __post_init__(self) -> None:
        if self.persistent != "int32":
            raise ValueError("persistent index storage is fixed to int32")
        if self.ephemeral not in ("auto", "int32", "int64"):
            raise ValueError(
                "ephemeral index policy must be 'auto', 'int32', or 'int64'"
            )


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    mode: ExecutionMode = "fast"

    def __post_init__(self) -> None:
        if self.mode not in ("fast", "deterministic"):
            raise ValueError("execution mode must be 'fast' or 'deterministic'")
        if self.mode == "deterministic":
            self.validate_global_state()

    def validate_global_state(self) -> None:
        if self.mode == "deterministic" and not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "deterministic execution requires "
                "torch.use_deterministic_algorithms(True); the library does not "
                "change process-wide PyTorch state"
            )


def resolve_execution_policy(
    policy: ExecutionPolicy | ExecutionMode | None,
) -> ExecutionPolicy:
    if policy is None:
        return ExecutionPolicy("fast")
    if isinstance(policy, ExecutionPolicy):
        policy.validate_global_state()
        return policy
    return ExecutionPolicy(policy)


@dataclass(frozen=True, slots=True)
class PackConfig:
    device: torch.device | str | None = None
    value_dtype: torch.dtype | None = None
    index_policy: IndexPolicy = field(default_factory=IndexPolicy)
    pin_cpu_staging: bool = False

    def __post_init__(self) -> None:
        if self.device is not None:
            object.__setattr__(self, "device", torch.device(self.device))
        if self.value_dtype is not None:
            probe = torch.empty((), dtype=self.value_dtype)
            if not probe.is_floating_point():
                raise TypeError("value_dtype must be a floating-point torch dtype")

    @classmethod
    def cuda_fp32(
        cls,
        device: torch.device | str = "cuda",
        *,
        index_policy: IndexPolicy | None = None,
        pin_cpu_staging: bool = False,
    ) -> "PackConfig":
        return cls(
            device=device,
            value_dtype=torch.float32,
            index_policy=index_policy or IndexPolicy(),
            pin_cpu_staging=pin_cpu_staging,
        )

    @classmethod
    def cuda_bf16(
        cls,
        device: torch.device | str = "cuda",
        *,
        index_policy: IndexPolicy | None = None,
        pin_cpu_staging: bool = False,
    ) -> "PackConfig":
        return cls(
            device=device,
            value_dtype=torch.bfloat16,
            index_policy=index_policy or IndexPolicy(),
            pin_cpu_staging=pin_cpu_staging,
        )

    @classmethod
    def cuda_fp16(
        cls,
        device: torch.device | str = "cuda",
        *,
        index_policy: IndexPolicy | None = None,
        pin_cpu_staging: bool = False,
    ) -> "PackConfig":
        return cls(
            device=device,
            value_dtype=torch.float16,
            index_policy=index_policy or IndexPolicy(),
            pin_cpu_staging=pin_cpu_staging,
        )


@dataclass(frozen=True, slots=True)
class ResolvedPackConfig:
    device: torch.device
    value_dtype: torch.dtype
    index_policy: IndexPolicy
    pin_cpu_staging: bool
    execution_mode: ExecutionMode

    @property
    def execution_policy(self) -> ExecutionPolicy:
        return ExecutionPolicy(self.execution_mode)

    @property
    def staging_policy(self) -> Literal["direct", "pinned_cpu"]:
        return "pinned_cpu" if self.pin_cpu_staging else "direct"


def _validate_explicit_bf16(device: torch.device, dtype: torch.dtype) -> None:
    if device.type != "cuda" or dtype != torch.bfloat16:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA BF16 was requested for {device}, but CUDA is unavailable"
        )
    try:
        with torch.cuda.device(device):
            supported = torch.cuda.is_bf16_supported()
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeError(f"cannot query BF16 support for {device}") from exc
    if not supported:
        raise RuntimeError(
            f"CUDA BF16 is not supported on {device}; no FP16 fallback is performed"
        )


def resolve_pack_config(
    pack_config: PackConfig | None,
    *,
    source_device: torch.device | None,
    source_dtype: torch.dtype | None,
    execution_policy: ExecutionPolicy | ExecutionMode | None,
    fallback_device: torch.device | str | None = None,
    fallback_dtype: torch.dtype | None = None,
) -> ResolvedPackConfig:
    policy = resolve_execution_policy(execution_policy)
    config = pack_config or PackConfig()

    device = config.device or source_device
    dtype = config.value_dtype or source_dtype
    if device is None:
        device = torch.device(fallback_device or "cpu")
    else:
        device = torch.device(device)
    if dtype is None:
        dtype = fallback_dtype or torch.float32

    if config.pin_cpu_staging and device.type == "cpu":
        raise ValueError("pin_cpu_staging=True is invalid for a CPU target")
    _validate_explicit_bf16(device, dtype)
    return ResolvedPackConfig(
        device=device,
        value_dtype=dtype,
        index_policy=config.index_policy,
        pin_cpu_staging=config.pin_cpu_staging,
        execution_mode=policy.mode,
    )


def cuda_fp32(
    device: torch.device | str = "cuda", *, pin_cpu_staging: bool = False
) -> PackConfig:
    return PackConfig.cuda_fp32(device, pin_cpu_staging=pin_cpu_staging)


def cuda_bf16(
    device: torch.device | str = "cuda", *, pin_cpu_staging: bool = False
) -> PackConfig:
    return PackConfig.cuda_bf16(device, pin_cpu_staging=pin_cpu_staging)


def cuda_fp16(
    device: torch.device | str = "cuda", *, pin_cpu_staging: bool = False
) -> PackConfig:
    return PackConfig.cuda_fp16(device, pin_cpu_staging=pin_cpu_staging)


cuda_fp32_pack_config = cuda_fp32
cuda_bf16_pack_config = cuda_bf16
cuda_fp16_pack_config = cuda_fp16
