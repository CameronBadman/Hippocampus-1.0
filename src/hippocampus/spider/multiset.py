from __future__ import annotations

import torch
from torch import nn

from .set_attention import PositionFreeCrossAttention, masked_mean
from .types import PaddedSet


class IdentityBiasedResidual(nn.Module):
    def __init__(self, d_model: int, *, initial_logit: float = -2.0) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.full((d_model,), initial_logit))

    def forward(self, value: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return value + torch.sigmoid(self.logit) * update


class CrossSetRead(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = PositionFreeCrossAttention(
            d_model,
            num_heads,
            dropout,
        )
        self.residual = IdentityBiasedResidual(d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def update(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        memory: PaddedSet,
    ) -> torch.Tensor:
        return self.attention(
            self.query_norm(query),
            query_mask,
            self.memory_norm(memory.values),
            memory.mask,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        memory: PaddedSet,
    ) -> torch.Tensor:
        updated = self.residual(
            query,
            self.update(query, query_mask, memory),
        )
        return self.output_norm(updated) * query_mask.unsqueeze(-1)


class EdgeValueComposition(nn.Module):
    """Continuously mix a shared bank of unnamed low-rank transforms."""

    def __init__(
        self,
        d_model: int,
        *,
        transform_count: int,
        adapter_rank: int,
    ) -> None:
        super().__init__()
        self.transform_count = transform_count
        self.down = nn.Parameter(
            torch.empty(transform_count, d_model, adapter_rank)
        )
        self.up = nn.Parameter(
            torch.empty(transform_count, adapter_rank, d_model)
        )
        self.mixture = nn.Linear(d_model, transform_count)
        self.scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.xavier_uniform_(self.down)
        nn.init.zeros_(self.up)

    def forward(
        self,
        content_update: torch.Tensor,
        edge: PaddedSet,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_summary = masked_mean(edge.values, edge.mask)
        weights = torch.softmax(self.mixture(edge_summary), dim=-1)
        low_rank = torch.einsum(
            "bpd,kdr->bkpr",
            content_update,
            self.down,
        )
        transformed = torch.einsum(
            "bkpr,krd->bkpd",
            low_rank,
            self.up,
        )
        mixed = torch.einsum("bk,bkpd->bpd", weights, transformed)
        result = content_update + torch.sigmoid(self.scale) * mixed
        return result, weights


class MultiSetSpiderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float,
        edge_mode: str,
        edge_transforms: int,
        adapter_rank: int,
    ) -> None:
        super().__init__()
        self.query_read = CrossSetRead(d_model, num_heads, dropout)
        self.evidence_read = CrossSetRead(d_model, num_heads, dropout)
        self.source_read = CrossSetRead(d_model, num_heads, dropout)
        self.edge_read = CrossSetRead(d_model, num_heads, dropout)
        self.destination_read = CrossSetRead(d_model, num_heads, dropout)
        self.self_read = CrossSetRead(d_model, num_heads, dropout)
        self.composition = (
            EdgeValueComposition(
                d_model,
                transform_count=edge_transforms,
                adapter_rank=adapter_rank,
            )
            if edge_mode == "compositional"
            else None
        )

    def forward(
        self,
        path: torch.Tensor,
        path_mask: torch.Tensor,
        query: PaddedSet,
        evidence: PaddedSet,
        source: PaddedSet,
        edge: PaddedSet,
        destination: PaddedSet,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        path = self.query_read(path, path_mask, query)
        path = self.evidence_read(path, path_mask, evidence)
        path = self.source_read(path, path_mask, source)
        path = self.edge_read(path, path_mask, edge)
        destination_update = self.destination_read.update(
            path,
            path_mask,
            destination,
        )
        mixture_weights = None
        if self.composition is not None:
            destination_update, mixture_weights = self.composition(
                destination_update,
                edge,
            )
        path = self.destination_read.output_norm(
            self.destination_read.residual(path, destination_update)
        )
        path = path * path_mask.unsqueeze(-1)
        path_set = PaddedSet(path, path_mask)
        path = self.self_read(path, path_mask, path_set)
        return path, mixture_weights
