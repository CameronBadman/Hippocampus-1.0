from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..manifold import PackedManifoldFamily
from .types import PaddedSet


@dataclass(frozen=True, slots=True)
class AttentionBackendStatus:
    padded_sdpa: bool
    cuda_varlen: bool
    reason: str


def attention_backend_status() -> AttentionBackendStatus:
    return AttentionBackendStatus(
        padded_sdpa=True,
        cuda_varlen=False,
        reason=(
            "Spider v0 uses the verified padded SDPA path; no stable public "
            "PyTorch packed-varlen SDPA contract is enabled"
        ),
    )


def padded_family_gather(
    family: PackedManifoldFamily,
    owner_ids: torch.Tensor,
    *,
    validate_ids: bool = True,
) -> PaddedSet:
    ids = owner_ids.to(device=family.device)
    gathered = family.gather(ids, validate_ids=validate_ids)
    layout = gathered.layout("single")
    selection_count = int(ids.numel())
    max_rows = layout.max_seqlen
    values = family.values.new_zeros(
        (selection_count, max_rows, family.width)
    )
    mask = torch.zeros(
        (selection_count, max_rows),
        dtype=torch.bool,
        device=family.device,
    )
    presence = (
        None
        if family.presence is None
        else family.presence.new_zeros((selection_count, max_rows))
    )
    if layout.batch_size:
        positions = layout.selection_positions.to(torch.int64)
        values = values.index_copy(0, positions, layout.values)
        mask = mask.index_copy(0, positions, layout.mask)
        if presence is not None and layout.presence is not None:
            presence = presence.index_copy(0, positions, layout.presence)
    return PaddedSet(values=values, mask=mask, presence=presence)


def project_padded_set(values: PaddedSet, projection: nn.Module) -> PaddedSet:
    projected = projection(values.values)
    if values.presence is not None:
        projected = projected * values.presence.unsqueeze(-1)
    projected = projected * values.mask.unsqueeze(-1)
    return PaddedSet(projected, values.mask, values.presence)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[1] == 0:
        return values.new_zeros((values.shape[0], values.shape[2]))
    weights = mask.to(values.dtype).unsqueeze(-1)
    total = (values * weights).sum(dim=1)
    return total / weights.sum(dim=1).clamp_min(1.0)


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[1] == 0:
        return values.new_zeros((values.shape[0], values.shape[2]))
    minimum = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask.unsqueeze(-1), minimum)
    result = masked.max(dim=1).values
    return torch.where(mask.any(dim=1, keepdim=True), result, torch.zeros_like(result))


class PositionFreeCrossAttention(nn.Module):
    """Multi-head SDPA without positional or causal information."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, rows, _ = value.shape
        return value.reshape(
            batch,
            rows,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        key_value: torch.Tensor,
        key_value_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape[0] == 0:
            return torch.zeros_like(query)
        if key_value.shape[1] == 0:
            return torch.zeros_like(query)
        active = key_value_mask.any(dim=1) & query_mask.any(dim=1)
        if not bool(active.any().item()):
            return torch.zeros_like(query)
        indices = torch.nonzero(active, as_tuple=False).flatten()
        q = self._heads(self.query(query[indices]))
        k = self._heads(self.key(key_value[indices]))
        v = self._heads(self.value(key_value[indices]))
        allowed = key_value_mask[indices, None, None, :]
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            indices.numel(),
            query.shape[1],
            self.d_model,
        )
        attended = self.output(attended)
        attended = attended * query_mask[indices].unsqueeze(-1)
        result = torch.zeros_like(query)
        return result.index_copy(0, indices, attended)
