from __future__ import annotations

import torch
from torch import nn

from .multiset import CrossSetRead
from .types import PaddedSet


def grouped_padded_messages(
    messages: torch.Tensor,
    graph_ids: torch.Tensor,
    *,
    graph_count: int,
) -> PaddedSet:
    if messages.shape[0] == 0:
        return PaddedSet(
            messages.new_empty((graph_count, 0, messages.shape[-1])),
            torch.empty(
                (graph_count, 0),
                dtype=torch.bool,
                device=messages.device,
            ),
        )
    owners = graph_ids.to(torch.int64)
    order = torch.argsort(owners, stable=True)
    grouped_owners = owners[order]
    counts = torch.bincount(grouped_owners, minlength=graph_count)
    max_rows = int(counts.max().item())
    starts = torch.cumsum(counts, dim=0) - counts
    ranks = torch.arange(
        messages.shape[0],
        device=messages.device,
        dtype=torch.int64,
    ) - torch.repeat_interleave(starts, counts)
    flat_positions = grouped_owners * max_rows + ranks
    flat_values = messages.new_zeros((graph_count * max_rows, messages.shape[-1]))
    flat_values = flat_values.index_copy(0, flat_positions, messages[order])
    flat_mask = torch.zeros(
        graph_count * max_rows,
        dtype=torch.bool,
        device=messages.device,
    ).index_fill(0, flat_positions, True)
    return PaddedSet(
        flat_values.reshape(graph_count, max_rows, messages.shape[-1]),
        flat_mask.reshape(graph_count, max_rows),
    )


class EvidenceUpdater(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.read = CrossSetRead(d_model, num_heads, dropout)

    def forward(
        self,
        evidence: torch.Tensor,
        messages: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        message_set = grouped_padded_messages(
            messages,
            graph_ids,
            graph_count=evidence.shape[0],
        )
        evidence_mask = torch.ones(
            evidence.shape[:2],
            dtype=torch.bool,
            device=evidence.device,
        )
        return self.read(evidence, evidence_mask, message_set)
