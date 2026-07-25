from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .types import CandidateOutputs


class CandidatePolicyHeads(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 7),
        )

    def forward(self, path_state: torch.Tensor) -> CandidateOutputs:
        pooled = path_state.mean(dim=1)
        raw = self.network(pooled)
        return CandidateOutputs(
            next_path_state=path_state,
            priority_logits=raw[:, 0],
            expand_logits=raw[:, 1],
            context_logits=raw[:, 2],
            evidence_logits=raw[:, 3],
            remaining_cost=F.softplus(raw[:, 4]),
            support_logits=raw[:, 5],
            conflict_logits=raw[:, 6],
        )

    def empty(
        self,
        *,
        reference: torch.Tensor,
        path_rows: int,
        d_model: int,
    ) -> CandidateOutputs:
        empty_state = reference.new_empty((0, path_rows, d_model))
        empty = reference.new_empty((0,))
        return CandidateOutputs(
            empty_state,
            empty,
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
        )
