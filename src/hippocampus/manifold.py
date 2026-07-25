from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch

from ._validation import (
    as_id_tensor,
    require_floating,
    require_int32_capacity,
    require_tensor,
    tensor_scalar_bool,
    validate_pointer,
)
from .config import IndexPolicy, ResolvedPackConfig

if TYPE_CHECKING:
    from .layouts import (
        ManifoldLayout,
        ManifoldLayoutConfig,
        ManifoldLayoutPolicy,
    )


@dataclass(frozen=True, slots=True)
class RaggedManifoldComponents:
    values: torch.Tensor
    offsets: torch.Tensor | Sequence[int]
    presence: torch.Tensor | None = None

    def __post_init__(self) -> None:
        require_tensor(self.values, "values", ndim=2)
        require_floating(self.values, "values")
        if self.presence is not None:
            require_tensor(self.presence, "presence", ndim=1)
            require_floating(self.presence, "presence")
            if self.presence.shape[0] != self.values.shape[0]:
                raise ValueError("presence must align with values rows")

    @property
    def owner_count(self) -> int:
        if isinstance(self.offsets, torch.Tensor):
            if self.offsets.ndim != 1:
                raise ValueError("offsets must be rank 1")
            return int(self.offsets.numel() - 1)
        return len(self.offsets) - 1

    @property
    def width(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class DenseCandidateComponents:
    values: torch.Tensor
    valid_mask: torch.Tensor
    presence: torch.Tensor | None = None

    def __post_init__(self) -> None:
        require_tensor(self.values, "values", ndim=3)
        require_floating(self.values, "values")
        require_tensor(self.valid_mask, "valid_mask", ndim=2, dtype=torch.bool)
        if tuple(self.valid_mask.shape) != tuple(self.values.shape[:2]):
            raise ValueError("valid_mask must match values[owners, slots]")
        if self.presence is not None:
            require_tensor(self.presence, "presence", ndim=2)
            require_floating(self.presence, "presence")
            if tuple(self.presence.shape) != tuple(self.values.shape[:2]):
                raise ValueError("presence must match values[owners, slots]")

    @property
    def owner_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def max_slots(self) -> int:
        return int(self.values.shape[1])

    @property
    def width(self) -> int:
        return int(self.values.shape[2])


@dataclass(frozen=True, slots=True)
class RaggedManifoldBatch:
    values: torch.Tensor
    offsets: torch.Tensor
    lengths: torch.Tensor
    owner_ids: torch.Tensor
    selection_positions: torch.Tensor
    owner_graph_ids: torch.Tensor
    row_owner_ids: torch.Tensor
    row_selection_positions: torch.Tensor
    base_source_row_ids: torch.Tensor
    forward_shuffle: torch.Tensor
    inverse_shuffle: torch.Tensor
    source_row_ids: torch.Tensor
    presence: torch.Tensor | None = None
    candidate_slot_ids: torch.Tensor | None = None

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    @property
    def owner_count(self) -> int:
        return int(self.owner_ids.numel())

    @property
    def total_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def source_graph_ids(self) -> torch.Tensor:
        return self.owner_graph_ids

    @property
    def candidate_dense_indices(self) -> torch.Tensor | None:
        if self.candidate_slot_ids is None:
            return None
        return torch.stack((self.row_owner_ids, self.candidate_slot_ids), dim=1)

    @property
    def dense_slot_ids(self) -> torch.Tensor | None:
        return self.candidate_slot_ids

    def validate(self) -> "RaggedManifoldBatch":
        require_tensor(self.values, "values", ndim=2)
        require_floating(self.values, "values")
        device = self.values.device
        require_tensor(
            self.offsets,
            "offsets",
            ndim=1,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        validate_pointer(self.offsets, "offsets", expected_final=self.total_rows)
        for name in ("lengths", "owner_ids", "owner_graph_ids"):
            require_tensor(
                getattr(self, name),
                name,
                ndim=1,
                dtype=torch.int32,
                device=device,
                contiguous=True,
            )
        for name in (
            "selection_positions",
            "row_selection_positions",
            "base_source_row_ids",
            "forward_shuffle",
            "inverse_shuffle",
            "source_row_ids",
        ):
            require_tensor(
                getattr(self, name),
                name,
                ndim=1,
                dtype=torch.int64,
                device=device,
                contiguous=True,
            )
        if self.offsets.numel() != self.owner_count + 1:
            raise ValueError("offsets must contain one entry per selected owner plus one")
        if self.lengths.numel() != self.owner_count:
            raise ValueError("lengths must align with selected owners")
        if self.selection_positions.numel() != self.owner_count:
            raise ValueError("selection_positions must align with selected owners")
        if self.owner_graph_ids.numel() != self.owner_count:
            raise ValueError("owner_graph_ids must align with selected owners")
        expected_lengths = self.offsets[1:] - self.offsets[:-1]
        if not torch.equal(self.lengths, expected_lengths):
            raise ValueError("lengths disagree with offsets")
        for name in (
            "row_owner_ids",
            "row_selection_positions",
            "base_source_row_ids",
            "forward_shuffle",
            "inverse_shuffle",
            "source_row_ids",
        ):
            if getattr(self, name).numel() != self.total_rows:
                raise ValueError(f"{name} must align with gathered rows")
        if self.presence is not None:
            require_tensor(
                self.presence, "presence", ndim=1, device=device, contiguous=True
            )
            if self.presence.numel() != self.total_rows:
                raise ValueError("presence must align with gathered rows")
        if self.candidate_slot_ids is not None:
            require_tensor(
                self.candidate_slot_ids,
                "candidate_slot_ids",
                ndim=1,
                dtype=torch.int32,
                device=device,
                contiguous=True,
            )
            if self.candidate_slot_ids.numel() != self.total_rows:
                raise ValueError("candidate_slot_ids must align with gathered rows")
        if self.total_rows:
            identity = torch.arange(
                self.total_rows, dtype=torch.int64, device=device
            )
            if not torch.equal(self.forward_shuffle[self.inverse_shuffle], identity):
                raise ValueError("forward_shuffle and inverse_shuffle are not inverses")
            if not torch.equal(
                self.source_row_ids,
                self.base_source_row_ids[self.forward_shuffle],
            ):
                raise ValueError("source_row_ids disagree with forward_shuffle")
        return self

    def layout(
        self,
        policy: "ManifoldLayoutPolicy" = "auto",
        *,
        supports_cuda_varlen: bool = False,
        config: "ManifoldLayoutConfig | None" = None,
    ) -> "ManifoldLayout":
        from .layouts import build_manifold_layout

        return build_manifold_layout(
            self,
            policy=policy,
            supports_cuda_varlen=supports_cuda_varlen,
            config=config,
        )


GatheredManifoldBatch = RaggedManifoldBatch


@dataclass(frozen=True, slots=True)
class PackedManifoldFamily:
    values: torch.Tensor
    offsets: torch.Tensor
    row_owner_ids: torch.Tensor
    owner_graph_ids: torch.Tensor
    resolved_pack_config: ResolvedPackConfig
    _lengths: torch.Tensor
    presence: torch.Tensor | None = None
    candidate_slot_ids: torch.Tensor | None = None

    @classmethod
    def unsafe_from_packed(
        cls,
        *,
        values: torch.Tensor,
        offsets: torch.Tensor,
        row_owner_ids: torch.Tensor,
        owner_graph_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        presence: torch.Tensor | None = None,
        candidate_slot_ids: torch.Tensor | None = None,
        resolved_pack_config: ResolvedPackConfig | None = None,
    ) -> "PackedManifoldFamily":
        """Build a deeply validated, zero-copy manifold family."""

        if resolved_pack_config is None:
            resolved_pack_config = ResolvedPackConfig(
                device=values.device,
                value_dtype=values.dtype,
                index_policy=IndexPolicy(),
                pin_cpu_staging=False,
                execution_mode="fast",
            )
        if lengths is None:
            lengths = offsets[1:] - offsets[:-1]
        result = cls(
            values=values,
            offsets=offsets,
            row_owner_ids=row_owner_ids,
            owner_graph_ids=owner_graph_ids,
            resolved_pack_config=resolved_pack_config,
            _lengths=lengths,
            presence=presence,
            candidate_slot_ids=candidate_slot_ids,
        )
        result.validate()
        return result

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    @property
    def family_width(self) -> int:
        return self.width

    @property
    def owner_count(self) -> int:
        return int(self.offsets.numel() - 1)

    @property
    def total_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def lengths(self) -> torch.Tensor:
        # Kept cached provisionally; the public accessor is intentionally stable.
        return self._lengths

    @property
    def candidate_dense_indices(self) -> torch.Tensor | None:
        if self.candidate_slot_ids is None:
            return None
        return torch.stack((self.row_owner_ids, self.candidate_slot_ids), dim=1)

    @property
    def dense_slot_ids(self) -> torch.Tensor | None:
        return self.candidate_slot_ids

    def validate(self) -> "PackedManifoldFamily":
        require_tensor(self.values, "values", ndim=2, contiguous=True)
        require_floating(self.values, "values")
        device = self.values.device
        if device != self.resolved_pack_config.device:
            raise ValueError("values device disagrees with resolved_pack_config")
        if self.values.dtype != self.resolved_pack_config.value_dtype:
            raise TypeError("values dtype disagrees with resolved_pack_config")
        self.resolved_pack_config.execution_policy.validate_global_state()

        require_tensor(
            self.offsets,
            "offsets",
            ndim=1,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        validate_pointer(self.offsets, "offsets", expected_final=self.total_rows)
        require_int32_capacity("manifold owner count", self.owner_count)
        require_int32_capacity("manifold row count", self.total_rows)
        require_tensor(
            self.row_owner_ids,
            "row_owner_ids",
            ndim=1,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        require_tensor(
            self.owner_graph_ids,
            "owner_graph_ids",
            ndim=1,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        require_tensor(
            self._lengths,
            "lengths",
            ndim=1,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        if self.row_owner_ids.numel() != self.total_rows:
            raise ValueError("row_owner_ids must align with values")
        if self.owner_graph_ids.numel() != self.owner_count:
            raise ValueError("owner_graph_ids must align with owners")
        if self._lengths.numel() != self.owner_count:
            raise ValueError("lengths must align with owners")
        expected_lengths = self.offsets[1:] - self.offsets[:-1]
        if not torch.equal(self._lengths, expected_lengths):
            raise ValueError("cached lengths disagree with offsets")
        expected_owners = torch.repeat_interleave(
            torch.arange(self.owner_count, dtype=torch.int32, device=device),
            self._lengths.to(torch.int64),
        )
        if not torch.equal(self.row_owner_ids, expected_owners):
            raise ValueError("row_owner_ids disagree with offsets")
        if self.presence is not None:
            require_tensor(
                self.presence, "presence", ndim=1, device=device, contiguous=True
            )
            require_floating(self.presence, "presence")
            if self.presence.dtype != self.values.dtype:
                raise TypeError("presence dtype must match values dtype")
            if self.presence.numel() != self.total_rows:
                raise ValueError("presence must align with values")
        if self.candidate_slot_ids is not None:
            require_tensor(
                self.candidate_slot_ids,
                "candidate_slot_ids",
                ndim=1,
                dtype=torch.int32,
                device=device,
                contiguous=True,
            )
            if self.candidate_slot_ids.numel() != self.total_rows:
                raise ValueError("candidate_slot_ids must align with values")
            if self.candidate_slot_ids.numel() and tensor_scalar_bool(
                torch.any(self.candidate_slot_ids < 0)
            ):
                raise ValueError("candidate_slot_ids may not be negative")
        return self

    def gather(
        self,
        owner_ids: Any,
        *,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
        validate_ids: bool = True,
    ) -> RaggedManifoldBatch:
        selected = as_id_tensor(
            owner_ids,
            name="owner_ids",
            device=self.device,
            validate_ids=validate_ids,
            upper_bound=self.owner_count,
        )
        selected_long = selected.to(torch.int64)
        selected_i32 = selected.to(torch.int32)
        selection_count = int(selected.numel())
        positions = torch.arange(
            selection_count, dtype=torch.int64, device=self.device
        )
        lengths = self._lengths[selected_long]
        total_rows = int(lengths.to(torch.int64).sum().item())
        require_int32_capacity("total selected manifold rows", total_rows)
        offsets64 = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=self.device),
                torch.cumsum(lengths.to(torch.int64), dim=0),
            )
        )
        offsets = offsets64.to(torch.int32)

        row_selection_positions = torch.repeat_interleave(
            positions, lengths.to(torch.int64)
        )
        if total_rows:
            source_starts = self.offsets[selected_long].to(torch.int64)
            local_rows = (
                torch.arange(total_rows, dtype=torch.int64, device=self.device)
                - offsets64[row_selection_positions]
            )
            base_source_row_ids = (
                source_starts[row_selection_positions] + local_rows
            )
        else:
            base_source_row_ids = torch.empty(
                0, dtype=torch.int64, device=self.device
            )

        identity = torch.arange(
            total_rows, dtype=torch.int64, device=self.device
        )
        if shuffle:
            if (
                self.resolved_pack_config.execution_mode == "deterministic"
                and generator is None
            ):
                raise RuntimeError(
                    "deterministic row shuffling requires an explicit seeded generator"
                )
            if total_rows:
                random_keys = torch.rand(
                    total_rows, device=self.device, generator=generator
                )
                random_order = torch.argsort(random_keys, stable=True)
                forward_shuffle = random_order[
                    torch.argsort(
                        row_selection_positions[random_order], stable=True
                    )
                ]
            else:
                forward_shuffle = identity
        else:
            forward_shuffle = identity
        inverse_shuffle = torch.empty_like(forward_shuffle)
        if total_rows:
            inverse_shuffle.scatter_(0, forward_shuffle, identity)

        source_row_ids = base_source_row_ids[forward_shuffle]
        shuffled_positions = row_selection_positions[forward_shuffle]
        values = self.values[source_row_ids].contiguous()
        presence = (
            None
            if self.presence is None
            else self.presence[source_row_ids].contiguous()
        )
        candidate_slot_ids = (
            None
            if self.candidate_slot_ids is None
            else self.candidate_slot_ids[source_row_ids].contiguous()
        )
        row_owner_ids = self.row_owner_ids[source_row_ids].contiguous()
        result = RaggedManifoldBatch(
            values=values,
            offsets=offsets.contiguous(),
            lengths=lengths.contiguous(),
            owner_ids=selected_i32.contiguous(),
            selection_positions=positions,
            owner_graph_ids=self.owner_graph_ids[selected_long].contiguous(),
            row_owner_ids=row_owner_ids,
            row_selection_positions=shuffled_positions.contiguous(),
            base_source_row_ids=base_source_row_ids.contiguous(),
            forward_shuffle=forward_shuffle.contiguous(),
            inverse_shuffle=inverse_shuffle.contiguous(),
            source_row_ids=source_row_ids.contiguous(),
            presence=presence,
            candidate_slot_ids=candidate_slot_ids,
        )
        return result

    def layout(
        self,
        owner_ids: Any,
        policy: "ManifoldLayoutPolicy" = "auto",
        *,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
        validate_ids: bool = True,
        supports_cuda_varlen: bool = False,
        config: "ManifoldLayoutConfig | None" = None,
    ) -> "ManifoldLayout":
        return self.gather(
            owner_ids,
            shuffle=shuffle,
            generator=generator,
            validate_ids=validate_ids,
        ).layout(
            policy,
            supports_cuda_varlen=supports_cuda_varlen,
            config=config,
        )


def is_ragged_tuple(components: Any) -> bool:
    if not (
        isinstance(components, tuple)
        and len(components) in (2, 3)
        and isinstance(components[0], torch.Tensor)
        and components[0].ndim == 2
    ):
        return False
    offsets = components[1]
    if isinstance(offsets, torch.Tensor):
        return offsets.ndim == 1 and offsets.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        )
    if isinstance(offsets, Sequence):
        return all(isinstance(offset, int) and not isinstance(offset, bool) for offset in offsets)
    return False


def value_tensors(components: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(components, PackedManifoldFamily):
        tensors = [components.values]
        if components.presence is not None:
            tensors.append(components.presence)
        return tuple(tensors)
    if isinstance(components, (RaggedManifoldComponents, DenseCandidateComponents)):
        tensors = [components.values]
        if components.presence is not None:
            tensors.append(components.presence)
        return tuple(tensors)
    if isinstance(components, torch.Tensor):
        return (components,)
    if is_ragged_tuple(components):
        tensors = [components[0]]
        if len(components) == 3 and components[2] is not None:
            tensors.append(components[2])
        return tuple(tensors)
    if isinstance(components, Sequence):
        return tuple(value for value in components if isinstance(value, torch.Tensor))
    return ()


def _owned_value_copy(
    tensor: torch.Tensor,
    *,
    resolved: ResolvedPackConfig,
    name: str,
) -> torch.Tensor:
    require_tensor(tensor, name)
    require_floating(tensor, name)
    if (
        resolved.pin_cpu_staging
        and tensor.device.type == "cpu"
        and resolved.device.type == "cuda"
    ):
        staged = tensor.clone()
        if not staged.is_pinned():
            staged = staged.pin_memory()
        return staged.to(
            device=resolved.device,
            dtype=resolved.value_dtype,
            non_blocking=True,
        ).contiguous()
    moved = tensor.to(device=resolved.device, dtype=resolved.value_dtype)
    if moved is tensor:
        moved = tensor.clone()
    return moved.contiguous()


def _offset_tensor(
    offsets: torch.Tensor | Sequence[int],
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(offsets, torch.Tensor):
        if offsets.ndim != 1:
            raise ValueError(f"{name} must be rank 1")
        if offsets.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"{name} must use an integer dtype")
        return offsets.to(device=device, dtype=torch.int32).clone().contiguous()
    return torch.as_tensor(offsets, device=device, dtype=torch.int32).contiguous()


def _rows_to_ragged(
    rows: Sequence[torch.Tensor],
    *,
    owner_count: int,
    width: int,
) -> RaggedManifoldComponents:
    if len(rows) != owner_count:
        raise ValueError(f"expected {owner_count} owner tensors, got {len(rows)}")
    if not rows:
        values = torch.empty((0, width), dtype=torch.float32)
        return RaggedManifoldComponents(values, torch.zeros(1, dtype=torch.int32))
    normalised: list[torch.Tensor] = []
    lengths: list[int] = []
    for owner, raw in enumerate(rows):
        if not isinstance(raw, torch.Tensor):
            raise TypeError(f"owner row set {owner} must be a torch.Tensor")
        value = raw.unsqueeze(0) if raw.ndim == 1 else raw
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(
                f"owner row set {owner} must have shape [rows, {width}]"
            )
        normalised.append(value)
        lengths.append(int(value.shape[0]))
    values = torch.cat(normalised, dim=0)
    offsets = torch.tensor(
        [0, *torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device=values.device,
    )
    return RaggedManifoldComponents(values, offsets)


def normalise_manifold_components(
    components: Any,
    *,
    owner_count: int,
    width: int,
) -> RaggedManifoldComponents | DenseCandidateComponents | PackedManifoldFamily:
    if isinstance(
        components,
        (RaggedManifoldComponents, DenseCandidateComponents, PackedManifoldFamily),
    ):
        return components
    if isinstance(components, torch.Tensor):
        if components.ndim != 2:
            raise ValueError(
                "a bare manifold tensor must have shape [owners, width]; "
                "use RaggedManifoldComponents for variable row counts"
            )
        if tuple(components.shape) != (owner_count, width):
            raise ValueError(
                f"bare manifold tensor must have shape [{owner_count}, {width}]"
            )
        offsets = torch.arange(
            owner_count + 1, dtype=torch.int32, device=components.device
        )
        return RaggedManifoldComponents(components, offsets)
    if is_ragged_tuple(components):
        return RaggedManifoldComponents(
            values=components[0],
            offsets=components[1],
            presence=components[2] if len(components) == 3 else None,
        )
    if isinstance(components, Sequence):
        return _rows_to_ragged(
            components, owner_count=owner_count, width=width
        )
    raise TypeError(
        "manifold values must be RaggedManifoldComponents, "
        "DenseCandidateComponents, PackedManifoldFamily, a dense owner tensor, "
        "or a sequence of per-owner tensors"
    )


def pack_manifold_family(
    components: Any,
    *,
    owner_count: int,
    width: int,
    owner_graph_ids: torch.Tensor,
    resolved_pack_config: ResolvedPackConfig,
    allow_empty: bool,
    family_name: str,
    validate: bool = True,
) -> PackedManifoldFamily:
    normalised = normalise_manifold_components(
        components, owner_count=owner_count, width=width
    )
    resolved = resolved_pack_config

    if isinstance(normalised, PackedManifoldFamily):
        if normalised.owner_count != owner_count:
            raise ValueError(f"{family_name} owner count does not match topology")
        ragged = RaggedManifoldComponents(
            normalised.values,
            normalised.offsets,
            normalised.presence,
        )
        packed = pack_manifold_family(
            ragged,
            owner_count=owner_count,
            width=width,
            owner_graph_ids=owner_graph_ids,
            resolved_pack_config=resolved,
            allow_empty=allow_empty,
            family_name=family_name,
            validate=validate,
        )
        if normalised.candidate_slot_ids is None:
            return packed
        candidate_slots = normalised.candidate_slot_ids.to(
            device=resolved.device, dtype=torch.int32
        ).clone()
        packed = PackedManifoldFamily(
            values=packed.values,
            offsets=packed.offsets,
            row_owner_ids=packed.row_owner_ids,
            owner_graph_ids=packed.owner_graph_ids,
            resolved_pack_config=packed.resolved_pack_config,
            _lengths=packed.lengths,
            presence=packed.presence,
            candidate_slot_ids=candidate_slots,
        )
        if validate:
            packed.validate()
        return packed

    candidate_slot_ids: torch.Tensor | None = None
    if isinstance(normalised, DenseCandidateComponents):
        if normalised.owner_count != owner_count:
            raise ValueError(
                f"{family_name} dense candidate owner count must be {owner_count}"
            )
        if normalised.width != width:
            raise ValueError(
                f"{family_name} width must be {width}, got {normalised.width}"
            )
        dense_values = _owned_value_copy(
            normalised.values, resolved=resolved, name=f"{family_name}.values"
        )
        valid_mask = normalised.valid_mask.to(
            device=resolved.device, dtype=torch.bool
        ).clone()
        values = dense_values[valid_mask].contiguous()
        counts64 = valid_mask.sum(dim=1, dtype=torch.int64)
        offsets64 = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=resolved.device),
                torch.cumsum(counts64, dim=0),
            )
        )
        total_rows = int(offsets64[-1].item())
        require_int32_capacity(f"{family_name} row count", total_rows)
        offsets = offsets64.to(torch.int32)
        dense_slots = torch.arange(
            normalised.max_slots, device=resolved.device, dtype=torch.int32
        ).expand(owner_count, -1)
        candidate_slot_ids = dense_slots[valid_mask].contiguous()
        presence = (
            None
            if normalised.presence is None
            else _owned_value_copy(
                normalised.presence,
                resolved=resolved,
                name=f"{family_name}.presence",
            )[valid_mask].contiguous()
        )
    else:
        if normalised.owner_count != owner_count:
            raise ValueError(
                f"{family_name} offsets describe {normalised.owner_count} owners; "
                f"topology requires {owner_count}"
            )
        if normalised.width != width:
            raise ValueError(
                f"{family_name} width must be {width}, got {normalised.width}"
            )
        values = _owned_value_copy(
            normalised.values, resolved=resolved, name=f"{family_name}.values"
        )
        offsets = _offset_tensor(
            normalised.offsets,
            device=resolved.device,
            name=f"{family_name}.offsets",
        )
        presence = (
            None
            if normalised.presence is None
            else _owned_value_copy(
                normalised.presence,
                resolved=resolved,
                name=f"{family_name}.presence",
            )
        )

    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{family_name}.values must have shape [rows, {width}]")
    if offsets.numel() != owner_count + 1:
        raise ValueError(f"{family_name}.offsets must have owner_count + 1 entries")
    validate_pointer(
        offsets, f"{family_name}.offsets", expected_final=int(values.shape[0])
    )
    lengths = (offsets[1:] - offsets[:-1]).contiguous()
    if not allow_empty and owner_count and tensor_scalar_bool(torch.any(lengths == 0)):
        empty_owners = torch.nonzero(lengths == 0).flatten()
        first = int(empty_owners[0].item())
        raise ValueError(
            f"{family_name} owner {first} has an empty manifold, which the schema forbids"
        )
    row_owner_ids = torch.repeat_interleave(
        torch.arange(owner_count, dtype=torch.int32, device=resolved.device),
        lengths.to(torch.int64),
    ).contiguous()
    owner_graph_ids_copy = owner_graph_ids.to(
        device=resolved.device, dtype=torch.int32
    ).clone().contiguous()
    packed = PackedManifoldFamily(
        values=values,
        offsets=offsets,
        row_owner_ids=row_owner_ids,
        owner_graph_ids=owner_graph_ids_copy,
        resolved_pack_config=resolved,
        _lengths=lengths,
        presence=presence,
        candidate_slot_ids=candidate_slot_ids,
    )
    if validate:
        packed.validate()
    return packed
