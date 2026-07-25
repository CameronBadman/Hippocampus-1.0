from __future__ import annotations

import torch
from torch import nn

from .multiset import CrossSetRead
from .types import PaddedSet


class ContextRefiner(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.read = CrossSetRead(d_model, num_heads, dropout)

    def forward(
        self,
        path_state: torch.Tensor,
        context: PaddedSet,
    ) -> torch.Tensor:
        path_mask = torch.ones(
            path_state.shape[:2],
            dtype=torch.bool,
            device=path_state.device,
        )
        return self.read(path_state, path_mask, context)
