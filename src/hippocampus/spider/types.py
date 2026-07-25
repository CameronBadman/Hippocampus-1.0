from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PaddedSet:
    values: torch.Tensor
    mask: torch.Tensor
    presence: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def row_count(self) -> int:
        return int(self.values.shape[1])

    @property
    def width(self) -> int:
        return int(self.values.shape[2])


@dataclass(frozen=True, slots=True)
class CandidateOutputs:
    next_path_state: torch.Tensor
    priority_logits: torch.Tensor
    expand_logits: torch.Tensor
    context_logits: torch.Tensor
    evidence_logits: torch.Tensor
    remaining_cost: torch.Tensor
    support_logits: torch.Tensor
    conflict_logits: torch.Tensor

    @property
    def candidate_count(self) -> int:
        return int(self.priority_logits.numel())

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.next_path_state,
            self.priority_logits,
            self.expand_logits,
            self.context_logits,
            self.evidence_logits,
            self.remaining_cost,
            self.support_logits,
            self.conflict_logits,
        )

    def index_copy(
        self,
        indices: torch.Tensor,
        source: "CandidateOutputs",
    ) -> "CandidateOutputs":
        if source.candidate_count != int(indices.numel()):
            raise ValueError("replacement outputs must align with selected indices")

        def copied(target: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
            return target.index_copy(0, indices, replacement)

        return CandidateOutputs(
            *(copied(target, replacement) for target, replacement in zip(
                self.tensors(),
                source.tensors(),
                strict=True,
            ))
        )
