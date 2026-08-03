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


class _PairwiseSetSimilarity(nn.Module):
    """Learned row matching with symmetric reductions over valid pairs."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.left_projection = nn.Linear(d_model, d_model, bias=False)
        self.right_projection = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model**-0.5

    def forward(self, left: PaddedSet, right: PaddedSet) -> torch.Tensor:
        if left.batch_size != right.batch_size:
            raise ValueError("paired sets must have equal batch sizes")
        left_values = self.left_projection(left.values)
        right_values = self.right_projection(right.values)
        similarities = torch.einsum(
            "bid,bjd->bij",
            left_values,
            right_values,
        ) * self.scale
        valid = left.mask.unsqueeze(2) & right.mask.unsqueeze(1)
        flat = similarities.flatten(1)
        flat_valid = valid.flatten(1)
        has_pairs = flat_valid.any(dim=1)
        masked = torch.where(
            flat_valid,
            flat,
            torch.full_like(flat, -torch.inf),
        )
        maximum = masked.max(dim=1).values
        pair_count = flat_valid.sum(dim=1).clamp_min(1)
        log_mean_exp = torch.logsumexp(masked, dim=1) - pair_count.log()
        zeros = similarities.new_zeros((left.batch_size,))
        return torch.stack(
            (
                torch.where(has_pairs, maximum, zeros),
                torch.where(has_pairs, log_mean_exp, zeros),
            ),
            dim=-1,
        )


class PairwiseEvidenceReadout(nn.Module):
    """Evidence head exposing cross-manifold identity comparisons directly."""

    def __init__(self, d_model: int, control_width: int) -> None:
        super().__init__()
        self.query_edge = _PairwiseSetSimilarity(d_model)
        self.query_destination = _PairwiseSetSimilarity(d_model)
        self.edge_destination = _PairwiseSetSimilarity(d_model)
        input_width = d_model + 6 + control_width
        self.output = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        path_state: torch.Tensor,
        context: CandidateReadoutContext,
    ) -> torch.Tensor:
        matches = torch.cat(
            (
                self.query_edge(context.query, context.edge),
                self.query_destination(context.query, context.destination),
                self.edge_destination(context.edge, context.destination),
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                path_state.mean(dim=1),
                matches,
                context.controller_features,
            ),
            dim=-1,
        )
        return self.output(features).squeeze(-1)
