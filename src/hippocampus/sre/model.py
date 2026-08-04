from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..segmented import segment_max, segment_mean
from .packed import PackedSRERetrievalBatch


@dataclass(frozen=True, slots=True)
class SREModelConfig:
    input_width: int = 384
    feature_width: int = 82
    hidden_width: int = 128
    attention_heads: int = 4
    set_layers: int = 1
    dropout: float = 0.0
    canonical_temperature: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "input_width",
            "feature_width",
            "hidden_width",
            "attention_heads",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.set_layers < 0:
            raise ValueError("set_layers must be non-negative")
        if self.hidden_width % self.attention_heads:
            raise ValueError("hidden_width must be divisible by attention_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.canonical_temperature <= 0:
            raise ValueError("canonical_temperature must be positive")


@dataclass(frozen=True, slots=True)
class SRERetrievalOutput:
    scores: torch.Tensor
    null_scores: torch.Tensor
    canonical_scores: torch.Tensor
    canonical_query: torch.Tensor
    canonical_candidates: torch.Tensor
    candidate_states: torch.Tensor


class PackedSRECanonicalRetriever(nn.Module):
    """Position-free retriever over the repository's packed graph substrate.

    The query root is expanded through CSR to enumerate candidates. Candidate
    neighbourhoods are then expanded through the same topology and reduced
    with segmented tensor operations. No parallel Python graph is consulted.
    """

    def __init__(self, config: SREModelConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_width
        self.query_projection = nn.Sequential(
            nn.LayerNorm(config.input_width),
            nn.Linear(config.input_width, width),
        )
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(config.input_width),
            nn.Linear(config.input_width, width),
        )
        self.edge_projection = nn.Sequential(
            nn.LayerNorm(config.input_width),
            nn.Linear(config.input_width, width),
        )
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(config.feature_width),
            nn.Linear(config.feature_width, width),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(8 * width),
            nn.Linear(8 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        if config.set_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=config.attention_heads,
                dim_feedforward=4 * width,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.candidate_set = nn.TransformerEncoder(
                layer,
                num_layers=config.set_layers,
                enable_nested_tensor=False,
            )
        else:
            self.candidate_set = nn.Identity()
        self.score_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.null_head = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.canonical_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _single_row_values(values, owner_count: int, name: str) -> torch.Tensor:
        if values.total_rows != owner_count or not bool(torch.all(values.lengths == 1)):
            raise ValueError(f"{name} must contain exactly one row per owner")
        return values.values

    def forward(self, batch: PackedSRERetrievalBatch) -> SRERetrievalOutput:
        config = self.config
        if batch.graph.schema.summary_dim != config.input_width:
            raise ValueError("packed graph width disagrees with model configuration")
        if batch.candidate_features.shape[-1] != config.feature_width:
            raise ValueError("candidate feature width disagrees with model configuration")

        query_rows = batch.query.gather(
            torch.arange(batch.batch_size, device=batch.device),
            validate_ids=False,
        )
        query_values = segment_mean(
            query_rows.values,
            query_rows.offsets,
        ).values

        candidate_expansion = batch.graph.topology.expand_frontier(
            batch.root_node_ids,
            validate_ids=False,
        )
        occurrence_count = batch.batch_size * batch.pool_size
        candidates = batch.graph.summaries.gather(
            candidate_expansion.destination_node_ids,
            validate_ids=False,
        )
        candidate_values = self._single_row_values(
            candidates,
            occurrence_count,
            "candidate summaries",
        )
        retrieval_edges = batch.graph.edges.gather(
            candidate_expansion.edge_ids,
            validate_ids=False,
        )
        retrieval_edge_values = self._single_row_values(
            retrieval_edges,
            occurrence_count,
            "retrieval edges",
        )

        neighbor_expansion = batch.graph.topology.expand_frontier(
            candidate_expansion.destination_node_ids,
            validate_ids=False,
        )
        neighbor_summaries = batch.graph.summaries.gather(
            neighbor_expansion.destination_node_ids,
            validate_ids=False,
        )
        neighbor_edges = batch.graph.edges.gather(
            neighbor_expansion.edge_ids,
            validate_ids=False,
        )
        neighbor_values = self._single_row_values(
            neighbor_summaries,
            neighbor_expansion.total_arcs,
            "neighbor summaries",
        )
        neighbor_edge_values = self._single_row_values(
            neighbor_edges,
            neighbor_expansion.total_arcs,
            "neighbor edges",
        )
        neighbor_mean = segment_mean(
            neighbor_values,
            neighbor_expansion.arc_offsets,
        ).values
        neighbor_maximum = segment_max(
            neighbor_values,
            neighbor_expansion.arc_offsets,
        ).values
        neighbor_edge_mean = segment_mean(
            neighbor_edge_values,
            neighbor_expansion.arc_offsets,
        ).values

        graph_positions = candidate_expansion.frontier_positions
        query_projected = self.query_projection(query_values)
        candidate_projected = self.memory_projection(candidate_values)
        query_occurrences = query_projected[graph_positions]
        neighbor_projected = self.memory_projection(neighbor_mean)
        neighbor_max_projected = self.memory_projection(neighbor_maximum)
        edge_projected = self.edge_projection(
            retrieval_edge_values + neighbor_edge_mean
        )
        feature_projected = self.feature_projection(
            batch.candidate_features.reshape(occurrence_count, -1)
        )

        canonical_query = F.normalize(query_projected, dim=-1)
        canonical_candidates = F.normalize(candidate_projected, dim=-1).reshape(
            batch.batch_size,
            batch.pool_size,
            config.hidden_width,
        )
        canonical_scores = torch.einsum(
            "bd,bkd->bk",
            canonical_query,
            canonical_candidates,
        ) / config.canonical_temperature
        fused = self.fusion(
            torch.cat(
                (
                    query_occurrences,
                    candidate_projected,
                    query_occurrences * candidate_projected,
                    torch.abs(query_occurrences - candidate_projected),
                    neighbor_projected,
                    neighbor_max_projected,
                    edge_projected,
                    feature_projected,
                ),
                dim=-1,
            )
        ).reshape(batch.batch_size, batch.pool_size, config.hidden_width)
        states = self.candidate_set(fused)
        learned_scores = self.score_head(states).squeeze(-1)
        scores = learned_scores + self.canonical_scale * canonical_scores
        pooled_mean = states.mean(dim=1)
        pooled_maximum = states.max(dim=1).values
        null_scores = self.null_head(
            torch.cat((query_projected, pooled_mean, pooled_maximum), dim=-1)
        ).squeeze(-1)
        return SRERetrievalOutput(
            scores=scores,
            null_scores=null_scores,
            canonical_scores=canonical_scores,
            canonical_query=canonical_query,
            canonical_candidates=canonical_candidates,
            candidate_states=states,
        )
