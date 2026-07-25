from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from ._validation import (
    next_power_of_two,
    require_int32_capacity,
    require_tensor,
    tensor_scalar_bool,
    validate_pointer,
    validate_tensor_allocation,
)
from .manifold import RaggedManifoldBatch


ManifoldLayoutPolicy = Literal["auto", "varlen", "power_of_two", "single"]


@dataclass(frozen=True, slots=True)
class ManifoldLayoutConfig:
    max_rows_per_manifold: int | None = None
    target_padded_rows_per_launch: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_rows_per_manifold",
            "target_padded_rows_per_launch",
        ):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class VarlenManifoldBatch:
    values: torch.Tensor
    cu_seqlens: torch.Tensor
    lengths: torch.Tensor
    max_seqlen: int
    owner_ids: torch.Tensor
    selection_positions: torch.Tensor
    owner_graph_ids: torch.Tensor
    row_owner_ids: torch.Tensor
    row_selection_positions: torch.Tensor
    base_source_row_ids: torch.Tensor
    forward_shuffle: torch.Tensor
    inverse_shuffle: torch.Tensor
    source_row_ids: torch.Tensor
    empty_owner_ids: torch.Tensor
    empty_selection_positions: torch.Tensor
    empty_owner_graph_ids: torch.Tensor
    presence: torch.Tensor | None = None
    candidate_slot_ids: torch.Tensor | None = None

    @property
    def nonempty_count(self) -> int:
        return int(self.owner_ids.numel())

    @property
    def empty_count(self) -> int:
        return int(self.empty_owner_ids.numel())

    @property
    def total_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    @property
    def nonempty_owner_ids(self) -> torch.Tensor:
        return self.owner_ids

    @property
    def nonempty_selection_positions(self) -> torch.Tensor:
        return self.selection_positions

    @property
    def candidate_dense_indices(self) -> torch.Tensor | None:
        if self.candidate_slot_ids is None:
            return None
        return torch.stack((self.row_owner_ids, self.candidate_slot_ids), dim=1)

    def validate(self) -> "VarlenManifoldBatch":
        require_tensor(self.values, "values", ndim=2, contiguous=True)
        device = self.values.device
        require_tensor(
            self.cu_seqlens,
            "cu_seqlens",
            ndim=1,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        validate_pointer(
            self.cu_seqlens,
            "cu_seqlens",
            expected_final=self.total_rows,
        )
        require_tensor(
            self.lengths,
            "lengths",
            ndim=1,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        if self.cu_seqlens.numel() != self.nonempty_count + 1:
            raise ValueError("cu_seqlens must align with non-empty owners")
        if self.lengths.numel() != self.nonempty_count:
            raise ValueError("lengths must align with non-empty owners")
        if self.lengths.numel() and tensor_scalar_bool(torch.any(self.lengths <= 0)):
            raise ValueError("varlen lengths must be strictly positive")
        expected_max = (
            int(self.lengths.max().item()) if self.lengths.numel() else 0
        )
        if self.max_seqlen != expected_max:
            raise ValueError("max_seqlen disagrees with lengths")
        for name in ("owner_ids", "owner_graph_ids", "empty_owner_ids", "empty_owner_graph_ids"):
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
            "empty_selection_positions",
        ):
            require_tensor(
                getattr(self, name),
                name,
                ndim=1,
                dtype=torch.int64,
                device=device,
                contiguous=True,
            )
        if self.presence is not None:
            require_tensor(
                self.presence, "presence", ndim=1, device=device, contiguous=True
            )
            if self.presence.numel() != self.total_rows:
                raise ValueError("presence must align with compact values")
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
                raise ValueError("candidate slots must align with compact values")
        return self


@dataclass(frozen=True, slots=True)
class PaddedManifoldBucket:
    values: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    padded_rows_per_manifold: int
    owner_ids: torch.Tensor
    selection_positions: torch.Tensor
    owner_graph_ids: torch.Tensor
    source_row_ids: torch.Tensor
    row_owner_ids: torch.Tensor
    presence: torch.Tensor | None = None
    candidate_slot_ids: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[2])

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.mask

    @property
    def max_seqlen(self) -> int:
        return self.padded_rows_per_manifold

    @property
    def padded_row_count(self) -> int:
        return self.batch_size * self.padded_rows_per_manifold

    @property
    def candidate_dense_indices(self) -> torch.Tensor | None:
        if self.candidate_slot_ids is None:
            return None
        return torch.stack((self.row_owner_ids, self.candidate_slot_ids), dim=2)

    def validate(self) -> "PaddedManifoldBucket":
        require_tensor(self.values, "values", ndim=3, contiguous=True)
        device = self.values.device
        if self.values.shape[1] != self.padded_rows_per_manifold:
            raise ValueError("values padding dimension disagrees with metadata")
        require_tensor(
            self.mask,
            "mask",
            ndim=2,
            dtype=torch.bool,
            device=device,
            contiguous=True,
        )
        if tuple(self.mask.shape) != tuple(self.values.shape[:2]):
            raise ValueError("mask must align with padded value rows")
        require_tensor(
            self.lengths,
            "lengths",
            ndim=1,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        if self.lengths.numel() != self.batch_size:
            raise ValueError("lengths must align with bucket batch")
        if self.lengths.numel() and (
            tensor_scalar_bool(torch.any(self.lengths <= 0))
            or tensor_scalar_bool(
                torch.any(self.lengths > self.padded_rows_per_manifold)
            )
        ):
            raise ValueError("bucket lengths are outside the padded extent")
        for name in ("owner_ids", "owner_graph_ids"):
            require_tensor(
                getattr(self, name),
                name,
                ndim=1,
                dtype=torch.int32,
                device=device,
                contiguous=True,
            )
            if getattr(self, name).numel() != self.batch_size:
                raise ValueError(f"{name} must align with bucket batch")
        require_tensor(
            self.selection_positions,
            "selection_positions",
            ndim=1,
            dtype=torch.int64,
            device=device,
            contiguous=True,
        )
        require_tensor(
            self.source_row_ids,
            "source_row_ids",
            ndim=2,
            dtype=torch.int64,
            device=device,
            contiguous=True,
        )
        require_tensor(
            self.row_owner_ids,
            "row_owner_ids",
            ndim=2,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        if tuple(self.source_row_ids.shape) != tuple(self.mask.shape):
            raise ValueError("source_row_ids must align with padded rows")
        if tuple(self.row_owner_ids.shape) != tuple(self.mask.shape):
            raise ValueError("row_owner_ids must align with padded rows")
        if self.presence is not None:
            require_tensor(
                self.presence, "presence", ndim=2, device=device, contiguous=True
            )
            if tuple(self.presence.shape) != tuple(self.mask.shape):
                raise ValueError("presence must align with padded rows")
        if self.candidate_slot_ids is not None:
            require_tensor(
                self.candidate_slot_ids,
                "candidate_slot_ids",
                ndim=2,
                dtype=torch.int32,
                device=device,
                contiguous=True,
            )
            if tuple(self.candidate_slot_ids.shape) != tuple(self.mask.shape):
                raise ValueError("candidate slots must align with padded rows")
        return self


@dataclass(frozen=True, slots=True)
class PowerOfTwoManifoldBatch:
    buckets: tuple[PaddedManifoldBucket, ...]
    empty_owner_ids: torch.Tensor
    empty_selection_positions: torch.Tensor
    empty_owner_graph_ids: torch.Tensor

    @property
    def empty_count(self) -> int:
        return int(self.empty_owner_ids.numel())

    @property
    def nonempty_count(self) -> int:
        return sum(bucket.batch_size for bucket in self.buckets)

    def validate(self) -> "PowerOfTwoManifoldBatch":
        for bucket in self.buckets:
            bucket.validate()
        device = self.empty_owner_ids.device
        require_tensor(
            self.empty_owner_ids,
            "empty_owner_ids",
            ndim=1,
            dtype=torch.int32,
            contiguous=True,
        )
        require_tensor(
            self.empty_selection_positions,
            "empty_selection_positions",
            ndim=1,
            dtype=torch.int64,
            device=device,
            contiguous=True,
        )
        require_tensor(
            self.empty_owner_graph_ids,
            "empty_owner_graph_ids",
            ndim=1,
            dtype=torch.int32,
            device=device,
            contiguous=True,
        )
        if not (
            self.empty_owner_ids.numel()
            == self.empty_selection_positions.numel()
            == self.empty_owner_graph_ids.numel()
        ):
            raise ValueError("empty-owner mappings must align")
        return self


@dataclass(frozen=True, slots=True)
class SingleManifoldBatch:
    values: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    max_seqlen: int
    owner_ids: torch.Tensor
    selection_positions: torch.Tensor
    owner_graph_ids: torch.Tensor
    source_row_ids: torch.Tensor
    row_owner_ids: torch.Tensor
    empty_owner_ids: torch.Tensor
    empty_selection_positions: torch.Tensor
    empty_owner_graph_ids: torch.Tensor
    presence: torch.Tensor | None = None
    candidate_slot_ids: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[2])

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.mask

    @property
    def candidate_dense_indices(self) -> torch.Tensor | None:
        if self.candidate_slot_ids is None:
            return None
        return torch.stack((self.row_owner_ids, self.candidate_slot_ids), dim=2)

    def validate(self) -> "SingleManifoldBatch":
        bucket = PaddedManifoldBucket(
            values=self.values,
            mask=self.mask,
            lengths=self.lengths,
            padded_rows_per_manifold=self.max_seqlen,
            owner_ids=self.owner_ids,
            selection_positions=self.selection_positions,
            owner_graph_ids=self.owner_graph_ids,
            source_row_ids=self.source_row_ids,
            row_owner_ids=self.row_owner_ids,
            presence=self.presence,
            candidate_slot_ids=self.candidate_slot_ids,
        )
        bucket.validate()
        if not (
            self.empty_owner_ids.numel()
            == self.empty_selection_positions.numel()
            == self.empty_owner_graph_ids.numel()
        ):
            raise ValueError("empty-owner mappings must align")
        return self


ManifoldLayout: TypeAlias = (
    VarlenManifoldBatch | PowerOfTwoManifoldBatch | SingleManifoldBatch
)


def _validate_manifold_lengths(
    batch: RaggedManifoldBatch, config: ManifoldLayoutConfig
) -> None:
    if batch.lengths.numel() == 0:
        return
    maximum = int(batch.lengths.max().item())
    require_int32_capacity("maximum manifold sequence length", maximum)
    if (
        config.max_rows_per_manifold is not None
        and maximum > config.max_rows_per_manifold
    ):
        raise ValueError(
            f"manifold length {maximum} exceeds max_rows_per_manifold="
            f"{config.max_rows_per_manifold}"
        )


def _selection_partitions(
    batch: RaggedManifoldBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    nonempty = torch.nonzero(batch.lengths > 0).flatten().to(torch.int64)
    empty = torch.nonzero(batch.lengths == 0).flatten().to(torch.int64)
    return nonempty, empty


def _empty_mappings(
    batch: RaggedManifoldBatch, empty_positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch.owner_ids[empty_positions].contiguous(),
        batch.selection_positions[empty_positions].contiguous(),
        batch.owner_graph_ids[empty_positions].contiguous(),
    )


def to_varlen_layout(
    batch: RaggedManifoldBatch,
    *,
    config: ManifoldLayoutConfig | None = None,
) -> VarlenManifoldBatch:
    layout_config = config or ManifoldLayoutConfig()
    _validate_manifold_lengths(batch, layout_config)
    nonempty, empty = _selection_partitions(batch)
    lengths = batch.lengths[nonempty].contiguous()
    cu64 = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=batch.values.device),
            torch.cumsum(lengths.to(torch.int64), dim=0),
        )
    )
    total_rows = int(cu64[-1].item())
    require_int32_capacity("varlen cumulative row count", total_rows)
    max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
    require_int32_capacity("varlen max_seqlen", max_seqlen)
    empty_owner_ids, empty_positions, empty_graph_ids = _empty_mappings(
        batch, empty
    )
    result = VarlenManifoldBatch(
        values=batch.values,
        cu_seqlens=cu64.to(torch.int32),
        lengths=lengths,
        max_seqlen=max_seqlen,
        owner_ids=batch.owner_ids[nonempty].contiguous(),
        selection_positions=batch.selection_positions[nonempty].contiguous(),
        owner_graph_ids=batch.owner_graph_ids[nonempty].contiguous(),
        row_owner_ids=batch.row_owner_ids,
        row_selection_positions=batch.row_selection_positions,
        base_source_row_ids=batch.base_source_row_ids,
        forward_shuffle=batch.forward_shuffle,
        inverse_shuffle=batch.inverse_shuffle,
        source_row_ids=batch.source_row_ids,
        empty_owner_ids=empty_owner_ids,
        empty_selection_positions=empty_positions,
        empty_owner_graph_ids=empty_graph_ids,
        presence=batch.presence,
        candidate_slot_ids=batch.candidate_slot_ids,
    )
    return result


def _make_padded_bucket(
    batch: RaggedManifoldBatch,
    selection_indices: torch.Tensor,
    *,
    padded_rows: int,
) -> PaddedManifoldBucket:
    device = batch.values.device
    selection_indices = selection_indices.to(device=device, dtype=torch.int64)
    lengths = batch.lengths[selection_indices].contiguous()
    batch_size = int(selection_indices.numel())
    width = batch.width
    if batch_size and int(lengths.max().item()) > padded_rows:
        raise ValueError("padded extent may not truncate a manifold")

    validate_tensor_allocation(
        "padded manifold values",
        (batch_size, padded_rows, width),
        batch.values.dtype,
    )
    validate_tensor_allocation(
        "padded manifold mask",
        (batch_size, padded_rows),
        torch.bool,
    )
    validate_tensor_allocation(
        "padded manifold source mappings",
        (batch_size, padded_rows),
        torch.int64,
        additional_element_factor=2,
    )

    selected_offsets64 = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(lengths.to(torch.int64), dim=0),
        )
    )
    total_rows = int(selected_offsets64[-1].item())
    bucket_positions = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int64, device=device),
        lengths.to(torch.int64),
    )
    if total_rows:
        source_starts = batch.offsets[selection_indices].to(torch.int64)
        local_rows = (
            torch.arange(total_rows, dtype=torch.int64, device=device)
            - selected_offsets64[bucket_positions]
        )
        rows_in_gathered = source_starts[bucket_positions] + local_rows
        rows_in_padded = bucket_positions * padded_rows + local_rows
    else:
        rows_in_gathered = torch.empty(0, dtype=torch.int64, device=device)
        rows_in_padded = torch.empty(0, dtype=torch.int64, device=device)

    flat_values = batch.values.new_zeros((batch_size * padded_rows, width))
    flat_values = flat_values.index_copy(
        0, rows_in_padded, batch.values[rows_in_gathered]
    )
    values = flat_values.reshape(batch_size, padded_rows, width).contiguous()

    flat_mask = torch.zeros(
        batch_size * padded_rows, dtype=torch.bool, device=device
    )
    flat_mask = flat_mask.index_fill(0, rows_in_padded, True)
    mask = flat_mask.reshape(batch_size, padded_rows).contiguous()

    flat_source_ids = torch.full(
        (batch_size * padded_rows,), -1, dtype=torch.int64, device=device
    )
    flat_source_ids = flat_source_ids.index_copy(
        0, rows_in_padded, batch.source_row_ids[rows_in_gathered]
    )
    source_row_ids = flat_source_ids.reshape(
        batch_size, padded_rows
    ).contiguous()

    flat_owner_ids = torch.full(
        (batch_size * padded_rows,), -1, dtype=torch.int32, device=device
    )
    flat_owner_ids = flat_owner_ids.index_copy(
        0, rows_in_padded, batch.row_owner_ids[rows_in_gathered]
    )
    row_owner_ids = flat_owner_ids.reshape(
        batch_size, padded_rows
    ).contiguous()

    if batch.presence is None:
        presence = None
    else:
        flat_presence = batch.presence.new_zeros(batch_size * padded_rows)
        flat_presence = flat_presence.index_copy(
            0, rows_in_padded, batch.presence[rows_in_gathered]
        )
        presence = flat_presence.reshape(batch_size, padded_rows).contiguous()

    if batch.candidate_slot_ids is None:
        candidate_slot_ids = None
    else:
        flat_candidate_slots = torch.full(
            (batch_size * padded_rows,), -1, dtype=torch.int32, device=device
        )
        flat_candidate_slots = flat_candidate_slots.index_copy(
            0,
            rows_in_padded,
            batch.candidate_slot_ids[rows_in_gathered],
        )
        candidate_slot_ids = flat_candidate_slots.reshape(
            batch_size, padded_rows
        ).contiguous()

    return PaddedManifoldBucket(
        values=values,
        mask=mask,
        lengths=lengths,
        padded_rows_per_manifold=padded_rows,
        owner_ids=batch.owner_ids[selection_indices].contiguous(),
        selection_positions=batch.selection_positions[
            selection_indices
        ].contiguous(),
        owner_graph_ids=batch.owner_graph_ids[selection_indices].contiguous(),
        source_row_ids=source_row_ids,
        row_owner_ids=row_owner_ids,
        presence=presence,
        candidate_slot_ids=candidate_slot_ids,
    )


def to_power_of_two_layout(
    batch: RaggedManifoldBatch,
    *,
    config: ManifoldLayoutConfig | None = None,
) -> PowerOfTwoManifoldBatch:
    layout_config = config or ManifoldLayoutConfig()
    _validate_manifold_lengths(batch, layout_config)
    nonempty, empty = _selection_partitions(batch)
    empty_owner_ids, empty_positions, empty_graph_ids = _empty_mappings(
        batch, empty
    )
    if nonempty.numel() == 0:
        return PowerOfTwoManifoldBatch(
            buckets=(),
            empty_owner_ids=empty_owner_ids,
            empty_selection_positions=empty_positions,
            empty_owner_graph_ids=empty_graph_ids,
        )

    nonempty_positions = nonempty.tolist()
    lengths_by_position = {
        position: int(batch.lengths[position].item())
        for position in nonempty_positions
    }
    positions_by_power: dict[int, list[int]] = {}
    for position in nonempty_positions:
        padded = next_power_of_two(lengths_by_position[position])
        positions_by_power.setdefault(padded, []).append(position)

    target = layout_config.target_padded_rows_per_launch
    buckets: list[PaddedManifoldBucket] = []
    for padded in sorted(positions_by_power):
        positions = positions_by_power[padded]
        if target is not None and padded > target:
            for position in positions:
                exact_length = lengths_by_position[position]
                bucket = _make_padded_bucket(
                    batch,
                    torch.tensor([position], dtype=torch.int64, device=batch.values.device),
                    padded_rows=exact_length,
                )
                buckets.append(bucket)
            continue

        owners_per_launch = len(positions)
        if target is not None:
            owners_per_launch = max(1, target // padded)
        for start in range(0, len(positions), owners_per_launch):
            chunk = positions[start : start + owners_per_launch]
            bucket = _make_padded_bucket(
                batch,
                torch.tensor(chunk, dtype=torch.int64, device=batch.values.device),
                padded_rows=padded,
            )
            buckets.append(bucket)

    return PowerOfTwoManifoldBatch(
        buckets=tuple(buckets),
        empty_owner_ids=empty_owner_ids,
        empty_selection_positions=empty_positions,
        empty_owner_graph_ids=empty_graph_ids,
    )


def to_single_layout(
    batch: RaggedManifoldBatch,
    *,
    config: ManifoldLayoutConfig | None = None,
) -> SingleManifoldBatch:
    layout_config = config or ManifoldLayoutConfig()
    _validate_manifold_lengths(batch, layout_config)
    nonempty, empty = _selection_partitions(batch)
    empty_owner_ids, empty_positions, empty_graph_ids = _empty_mappings(
        batch, empty
    )
    if nonempty.numel() == 0:
        width = batch.width
        values = batch.values.new_empty((0, 0, width))
        mask = torch.empty(
            (0, 0), dtype=torch.bool, device=batch.values.device
        )
        source_ids = torch.empty(
            (0, 0), dtype=torch.int64, device=batch.values.device
        )
        row_owner_ids = torch.empty(
            (0, 0), dtype=torch.int32, device=batch.values.device
        )
        presence = (
            None
            if batch.presence is None
            else batch.presence.new_empty((0, 0))
        )
        candidate_slot_ids = (
            None
            if batch.candidate_slot_ids is None
            else torch.empty(
                (0, 0), dtype=torch.int32, device=batch.values.device
            )
        )
        return SingleManifoldBatch(
            values=values,
            mask=mask,
            lengths=torch.empty(
                0, dtype=torch.int32, device=batch.values.device
            ),
            max_seqlen=0,
            owner_ids=torch.empty(
                0, dtype=torch.int32, device=batch.values.device
            ),
            selection_positions=torch.empty(
                0, dtype=torch.int64, device=batch.values.device
            ),
            owner_graph_ids=torch.empty(
                0, dtype=torch.int32, device=batch.values.device
            ),
            source_row_ids=source_ids,
            row_owner_ids=row_owner_ids,
            empty_owner_ids=empty_owner_ids,
            empty_selection_positions=empty_positions,
            empty_owner_graph_ids=empty_graph_ids,
            presence=presence,
            candidate_slot_ids=candidate_slot_ids,
        )

    max_seqlen = int(batch.lengths[nonempty].max().item())
    padded_rows = int(nonempty.numel()) * max_seqlen
    target = layout_config.target_padded_rows_per_launch
    if target is not None and padded_rows > target:
        raise ValueError(
            f"single layout requires {padded_rows} padded rows, exceeding "
            f"target_padded_rows_per_launch={target}"
        )
    bucket = _make_padded_bucket(
        batch, nonempty, padded_rows=max_seqlen
    )
    return SingleManifoldBatch(
        values=bucket.values,
        mask=bucket.mask,
        lengths=bucket.lengths,
        max_seqlen=max_seqlen,
        owner_ids=bucket.owner_ids,
        selection_positions=bucket.selection_positions,
        owner_graph_ids=bucket.owner_graph_ids,
        source_row_ids=bucket.source_row_ids,
        row_owner_ids=bucket.row_owner_ids,
        empty_owner_ids=empty_owner_ids,
        empty_selection_positions=empty_positions,
        empty_owner_graph_ids=empty_graph_ids,
        presence=bucket.presence,
        candidate_slot_ids=bucket.candidate_slot_ids,
    )


def build_manifold_layout(
    batch: RaggedManifoldBatch,
    *,
    policy: ManifoldLayoutPolicy = "auto",
    supports_cuda_varlen: bool = False,
    config: ManifoldLayoutConfig | None = None,
) -> ManifoldLayout:
    if policy == "auto":
        policy = (
            "varlen"
            if batch.values.is_cuda and supports_cuda_varlen
            else "power_of_two"
        )
    if policy == "varlen":
        return to_varlen_layout(batch, config=config)
    if policy == "power_of_two":
        return to_power_of_two_layout(batch, config=config)
    if policy == "single":
        return to_single_layout(batch, config=config)
    raise ValueError(
        "layout policy must be 'auto', 'varlen', 'power_of_two', or 'single'"
    )


collate_manifolds = build_manifold_layout
