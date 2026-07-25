from __future__ import annotations

import torch
from torch import nn


TERMINATION_CLASS_COUNT = 6


class TerminationHead(nn.Module):
    def __init__(self, d_model: int, control_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(3 * d_model + control_width),
            nn.Linear(3 * d_model + control_width, d_model),
            nn.GELU(),
            nn.Linear(d_model, TERMINATION_CLASS_COUNT),
        )

    def forward(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        frontier: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat((query, evidence, frontier, control), dim=-1))
