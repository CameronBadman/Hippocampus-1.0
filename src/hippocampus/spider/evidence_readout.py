from __future__ import annotations

import torch
from torch import nn

from .multiset import CrossSetRead
from .types import CandidateReadoutContext, PaddedSet


class DedicatedPooledEvidenceReadout(nn.Module):
    """Evidence-only MLP over the historical mean-pooled path state."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        path_state: torch.Tensor,
        context: CandidateReadoutContext,
    ) -> torch.Tensor:
        del context
        return self.network(path_state.mean(dim=1)).squeeze(-1)


class SlotAwareEvidenceReadout(nn.Module):
    """One position-free evidence query reading every operational set."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        control_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.evidence_query = nn.Parameter(torch.empty(1, d_model))
        nn.init.normal_(self.evidence_query, std=0.02)
        self.control_projection = nn.Linear(control_width, d_model)
        self.path_read = CrossSetRead(d_model, num_heads, dropout)
        self.query_read = CrossSetRead(d_model, num_heads, dropout)
        self.source_read = CrossSetRead(d_model, num_heads, dropout)
        self.edge_read = CrossSetRead(d_model, num_heads, dropout)
        self.destination_read = CrossSetRead(d_model, num_heads, dropout)
        self.global_evidence_read = CrossSetRead(
            d_model,
            num_heads,
            dropout,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        path_state: torch.Tensor,
        context: CandidateReadoutContext,
    ) -> torch.Tensor:
        count = path_state.shape[0]
        token = self.evidence_query.unsqueeze(0).expand(count, -1, -1)
        token = token + self.control_projection(
            context.controller_features
        ).unsqueeze(1)
        token_mask = torch.ones(
            (count, 1),
            dtype=torch.bool,
            device=path_state.device,
        )
        path = PaddedSet(
            values=path_state,
            mask=torch.ones(
                path_state.shape[:2],
                dtype=torch.bool,
                device=path_state.device,
            ),
        )
        for read, values in (
            (self.path_read, path),
            (self.query_read, context.query),
            (self.source_read, context.source),
            (self.edge_read, context.edge),
            (self.destination_read, context.destination),
            (self.global_evidence_read, context.global_evidence),
        ):
            token = read(token, token_mask, values)
        return self.output(token[:, 0]).squeeze(-1)
