from __future__ import annotations

import torch
from torch import nn

from ..segmented import segment_max, segment_mean
from .types import CandidateOutputs


class CandidateEvidenceSetDecoder(nn.Module):
    """Predict the evidence count visible in each current candidate set.

    Candidate order is deliberately discarded by segmented mean/max pooling.
    The zero class is the current-set null action; this module does not predict
    total case evidence cardinality.
    """

    class_count = 5

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        candidate_width = d_model + 7
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(candidate_width),
            nn.Linear(candidate_width, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(2 * d_model + 2),
            nn.Linear(2 * d_model + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.class_count),
        )

    def forward(
        self,
        outputs: CandidateOutputs,
        candidate_graph_ids: torch.Tensor,
        *,
        graph_count: int,
    ) -> torch.Tensor:
        if graph_count < 0:
            raise ValueError("graph_count must be non-negative")
        owners = candidate_graph_ids.to(
            device=outputs.evidence_logits.device,
            dtype=torch.int64,
        )
        if owners.ndim != 1 or owners.numel() != outputs.candidate_count:
            raise ValueError("candidate graph IDs must align with candidates")
        if owners.numel() and (
            bool((owners < 0).any().item())
            or bool((owners >= graph_count).any().item())
        ):
            raise IndexError("candidate graph ID is out of range")
        scalars = torch.stack(
            (
                outputs.priority_logits,
                outputs.expand_logits,
                outputs.context_logits,
                outputs.evidence_logits,
                outputs.remaining_cost,
                outputs.support_logits,
                outputs.conflict_logits,
            ),
            dim=-1,
        )
        candidate_features = torch.cat(
            (outputs.next_path_state.mean(dim=1), scalars),
            dim=-1,
        )
        encoded = self.candidate_encoder(candidate_features)
        mean = segment_mean(
            encoded,
            row_owner_ids=owners,
            num_segments=graph_count,
        )
        maximum = segment_max(
            encoded,
            row_owner_ids=owners,
            num_segments=graph_count,
        )
        counts = torch.bincount(owners, minlength=graph_count).to(
            dtype=encoded.dtype
        )
        set_features = torch.cat(
            (
                mean.values,
                maximum.values,
                torch.log1p(counts).unsqueeze(-1),
                (counts == 0).to(encoded.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.output(set_features)


def candidate_evidence_count_targets(
    positive_mask: torch.Tensor,
    candidate_graph_ids: torch.Tensor,
    candidate_edge_ids: torch.Tensor,
    *,
    graph_count: int,
) -> torch.Tensor:
    """Count unique positive logical edges visible per graph, capped at 4+."""

    if graph_count < 0:
        raise ValueError("graph_count must be non-negative")
    if positive_mask.ndim != 1 or positive_mask.dtype is not torch.bool:
        raise TypeError("positive_mask must be a rank-1 bool tensor")
    if candidate_graph_ids.ndim != 1 or candidate_edge_ids.ndim != 1:
        raise ValueError("candidate IDs must be rank-1 tensors")
    if not (
        positive_mask.numel()
        == candidate_graph_ids.numel()
        == candidate_edge_ids.numel()
    ):
        raise ValueError("candidate count targets require aligned tensors")
    device = positive_mask.device
    owners = candidate_graph_ids.to(device=device, dtype=torch.int64)
    edges = candidate_edge_ids.to(device=device, dtype=torch.int64)
    if owners.numel() and (
        bool((owners < 0).any().item())
        or bool((owners >= graph_count).any().item())
    ):
        raise IndexError("candidate graph ID is out of range")
    positive_owners = owners[positive_mask]
    positive_edges = edges[positive_mask]
    if positive_edges.numel():
        order = torch.argsort(positive_edges, stable=True)
        sorted_edges = positive_edges[order]
        sorted_owners = positive_owners[order]
        first = torch.ones(
            sorted_edges.numel(),
            dtype=torch.bool,
            device=device,
        )
        first[1:] = sorted_edges[1:] != sorted_edges[:-1]
        positive_owners = sorted_owners[first]
    counts = torch.bincount(positive_owners, minlength=graph_count)
    return counts.clamp(max=4).to(torch.int64)
