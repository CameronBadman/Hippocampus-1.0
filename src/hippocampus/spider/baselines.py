from __future__ import annotations

import torch
from torch import nn

from ..programs.batching import PackedProgramBatch
from ..topology import FrontierExpansion
from .config import SpiderModelConfig
from .hypothesis import HypothesisBatch
from .model import CandidateScorerBase
from .multiset import IdentityBiasedResidual
from .set_attention import masked_max, masked_mean
from .types import CandidateOutputs, PaddedSet


def _symmetric_pool(values: PaddedSet) -> torch.Tensor:
    return torch.cat(
        (
            masked_mean(values.values, values.mask),
            masked_max(values.values, values.mask),
        ),
        dim=-1,
    )


class PooledScorer(CandidateScorerBase):
    """Symmetric mean/max MLP control with no manifold recurrence."""

    def __init__(self, config: SpiderModelConfig) -> None:
        super().__init__(config)
        family_count = 6
        self.transition = nn.Sequential(
            nn.LayerNorm(family_count * 2 * config.d_model + config.control_width),
            nn.Linear(
                family_count * 2 * config.d_model + config.control_width,
                2 * config.d_model,
            ),
            nn.GELU(),
            nn.Linear(2 * config.d_model, config.d_model),
        )
        self.residual = IdentityBiasedResidual(config.d_model)

    def score_candidates(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        expansion: FrontierExpansion,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
        *,
        round_index: int = 0,
    ) -> CandidateOutputs:
        del round_index
        self._validate_batch_widths(batch)
        if expansion.total_arcs == 0:
            return self.policy_heads.empty(
                reference=self.path_seed,
                path_rows=self.config.path_rows,
                d_model=self.config.d_model,
            )
        parent = expansion.frontier_positions.to(torch.int64)
        path = hypotheses.path_state[parent]
        query, source, edge, destination, evidence_set, _ = self._candidate_sets(
            batch,
            expansion,
            evidence,
        )
        path_set = PaddedSet(
            path,
            torch.ones(path.shape[:2], dtype=torch.bool, device=path.device),
        )
        control = (
            path.new_zeros((path.shape[0], self.config.control_width))
            if controller_features is None
            else controller_features
        )
        features = torch.cat(
            (
                _symmetric_pool(path_set),
                _symmetric_pool(query),
                _symmetric_pool(source),
                _symmetric_pool(edge),
                _symmetric_pool(destination),
                _symmetric_pool(evidence_set),
                control,
            ),
            dim=-1,
        )
        update = self.transition(features).unsqueeze(1)
        next_path = self.residual(path, update)
        return self.policy_heads(next_path)


class FlatTransformerScorer(CandidateScorerBase):
    """Position-free flat Transformer baseline with operational boundaries."""

    def __init__(self, config: SpiderModelConfig) -> None:
        super().__init__(config)
        self.family_embeddings = nn.Parameter(
            torch.empty(6, config.d_model)
        )
        nn.init.normal_(self.family_embeddings, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.num_heads,
                    dim_feedforward=4 * config.d_model,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.control_projection = nn.Linear(config.control_width, config.d_model)

    def _with_family(self, values: PaddedSet, family_id: int) -> PaddedSet:
        return PaddedSet(
            (
                values.values
                + self.family_embeddings[family_id].view(1, 1, -1)
            )
            * values.mask.unsqueeze(-1),
            values.mask,
        )

    def score_candidates(
        self,
        batch: PackedProgramBatch,
        hypotheses: HypothesisBatch,
        expansion: FrontierExpansion,
        evidence: torch.Tensor,
        controller_features: torch.Tensor | None = None,
        *,
        round_index: int = 0,
    ) -> CandidateOutputs:
        del round_index
        self._validate_batch_widths(batch)
        if expansion.total_arcs == 0:
            return self.policy_heads.empty(
                reference=self.path_seed,
                path_rows=self.config.path_rows,
                d_model=self.config.d_model,
            )
        parent = expansion.frontier_positions.to(torch.int64)
        path = hypotheses.path_state[parent]
        query, source, edge, destination, evidence_set, _ = self._candidate_sets(
            batch,
            expansion,
            evidence,
        )
        control = (
            path.new_zeros((path.shape[0], self.config.control_width))
            if controller_features is None
            else controller_features
        )
        path = path + self.control_projection(control).unsqueeze(1)
        path_set = PaddedSet(
            path,
            torch.ones(path.shape[:2], dtype=torch.bool, device=path.device),
        )
        families = (
            self._with_family(path_set, 0),
            self._with_family(query, 1),
            self._with_family(source, 2),
            self._with_family(edge, 3),
            self._with_family(destination, 4),
            self._with_family(evidence_set, 5),
        )
        values = torch.cat([family.values for family in families], dim=1)
        mask = torch.cat([family.mask for family in families], dim=1)
        for layer in self.layers:
            values = layer(values, src_key_padding_mask=~mask)
            values = values * mask.unsqueeze(-1)
        next_path = values[:, : self.config.path_rows]
        return self.policy_heads(next_path)
