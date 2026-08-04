from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..programs.batching import BindingTargets
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


class CanonicalBindingEvidenceReadout(nn.Module):
    """Evidence energy from aligned query/edge/destination row matches."""

    def __init__(self, d_model: int, control_width: int) -> None:
        super().__init__()
        if d_model <= 0 or control_width <= 0:
            raise ValueError("canonical binding dimensions must be positive")
        self.query_canonicalizer = nn.Linear(d_model, d_model)
        self.edge_canonicalizer = nn.Linear(d_model, d_model)
        self.summary_canonicalizer = nn.Linear(d_model, d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model + 6 + control_width),
            nn.Linear(d_model + 6 + control_width, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def canonicalizer_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for module in (
                self.query_canonicalizer,
                self.edge_canonicalizer,
                self.summary_canonicalizer,
            )
            for parameter in module.parameters()
        )

    @staticmethod
    def _canonical_set(values: PaddedSet, projection: nn.Module) -> PaddedSet:
        return PaddedSet(
            F.normalize(projection(values.values), dim=-1),
            values.mask,
            values.presence,
        )

    @staticmethod
    def _pair_statistics(left: PaddedSet, right: PaddedSet) -> torch.Tensor:
        if left.batch_size != right.batch_size:
            raise ValueError("binding sets must have equal batch sizes")
        if left.row_count == 0 or right.row_count == 0:
            return left.values.new_zeros((left.batch_size, 2))
        similarity = torch.einsum(
            "bid,bjd->bij",
            left.values,
            right.values,
        )
        valid = left.mask.unsqueeze(2) & right.mask.unsqueeze(1)
        flat = similarity.flatten(1)
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
        zeros = flat.new_zeros((left.batch_size,))
        return torch.stack(
            (
                torch.where(has_pairs, maximum, zeros),
                torch.where(has_pairs, log_mean_exp, zeros),
            ),
            dim=-1,
        )

    def forward(
        self,
        path_state: torch.Tensor,
        context: CandidateReadoutContext,
    ) -> torch.Tensor:
        query = self._canonical_set(
            context.query,
            self.query_canonicalizer,
        )
        edge = self._canonical_set(context.edge, self.edge_canonicalizer)
        destination = self._canonical_set(
            context.destination,
            self.summary_canonicalizer,
        )
        query_edge = self._pair_statistics(query, edge)
        query_destination = self._pair_statistics(query, destination)
        best_edge = query_edge[:, 0]
        best_destination = query_destination[:, 0]
        conjunction = torch.stack(
            (
                torch.minimum(best_edge, best_destination),
                best_edge * best_destination,
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                path_state.mean(dim=1),
                query_edge,
                query_destination,
                conjunction,
                context.controller_features,
            ),
            dim=-1,
        )
        return self.output(features).squeeze(-1)

    @staticmethod
    def _multi_positive_infonce(
        left: torch.Tensor,
        right: torch.Tensor,
        pairs: torch.Tensor,
        *,
        temperature: float,
    ) -> tuple[torch.Tensor, int]:
        if pairs.numel() == 0:
            return left.sum() * 0.0, 0
        positive = torch.zeros(
            (left.shape[0], right.shape[0]),
            dtype=torch.bool,
            device=left.device,
        )
        positive[pairs[:, 0], pairs[:, 1]] = True
        active_left = positive.any(dim=1)
        active_right = positive.any(dim=0)
        positive = positive[active_left][:, active_right]
        logits = (
            F.normalize(left[active_left], dim=-1)
            @ F.normalize(right[active_right], dim=-1).T
        ) / temperature
        masked = torch.where(
            positive,
            logits,
            torch.full_like(logits, -torch.inf),
        )
        left_loss = (
            torch.logsumexp(logits, dim=1)
            - torch.logsumexp(masked, dim=1)
        ).mean()
        right_loss = (
            torch.logsumexp(logits, dim=0)
            - torch.logsumexp(masked, dim=0)
        ).mean()
        return 0.5 * (left_loss + right_loss), int(pairs.shape[0])

    def alignment_loss(
        self,
        query: torch.Tensor,
        summaries: torch.Tensor,
        edges: torch.Tensor,
        targets: BindingTargets,
        *,
        temperature: float,
    ) -> tuple[torch.Tensor, int]:
        if temperature <= 0:
            raise ValueError("binding temperature must be positive")
        canonical_query = self.query_canonicalizer(query)
        summary_loss, summary_count = self._multi_positive_infonce(
            canonical_query,
            self.summary_canonicalizer(summaries),
            targets.query_summary_pairs,
            temperature=temperature,
        )
        edge_loss, edge_count = self._multi_positive_infonce(
            canonical_query,
            self.edge_canonicalizer(edges),
            targets.query_edge_pairs,
            temperature=temperature,
        )
        active = int(summary_count > 0) + int(edge_count > 0)
        if active == 0:
            return canonical_query.sum() * 0.0, 0
        return (
            (summary_loss + edge_loss) / active,
            summary_count + edge_count,
        )
