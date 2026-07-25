from __future__ import annotations

import torch
from torch import nn

from .config import SpiderModelConfig
from .multiset import IdentityBiasedResidual, MultiSetSpiderBlock
from .types import PaddedSet


class ArcProcessor(nn.Module):
    def __init__(self, config: SpiderModelConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                MultiSetSpiderBlock(
                    config.d_model,
                    config.num_heads,
                    dropout=config.dropout,
                    edge_mode=config.edge_mode,
                    edge_transforms=config.edge_transforms,
                    adapter_rank=config.adapter_rank,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.control_projection = nn.Sequential(
            nn.Linear(config.control_width, config.d_model),
            nn.Tanh(),
        )
        self.control_residual = IdentityBiasedResidual(config.d_model)

    def forward(
        self,
        path: torch.Tensor,
        query: PaddedSet,
        source: PaddedSet,
        edge: PaddedSet,
        destination: PaddedSet,
        control: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        path_mask = torch.ones(
            path.shape[:2],
            dtype=torch.bool,
            device=path.device,
        )
        control_update = self.control_projection(control).unsqueeze(1)
        path = self.control_residual(path, control_update)
        mixtures: list[torch.Tensor] = []
        for block in self.blocks:
            path, weights = block(
                path,
                path_mask,
                query,
                source,
                edge,
                destination,
            )
            if weights is not None:
                mixtures.append(weights)
        return path, tuple(mixtures)
