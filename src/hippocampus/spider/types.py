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

    def index_select(self, indices: torch.Tensor) -> "PaddedSet":
        resolved = indices.to(device=self.values.device, dtype=torch.int64)
        return PaddedSet(
            values=self.values[resolved],
            mask=self.mask[resolved],
            presence=(
                None if self.presence is None else self.presence[resolved]
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateReadoutContext:
    query: PaddedSet
    source: PaddedSet
    edge: PaddedSet
    destination: PaddedSet
    global_evidence: PaddedSet
    controller_features: torch.Tensor

    def index_select(self, indices: torch.Tensor) -> "CandidateReadoutContext":
        resolved = indices.to(
            device=self.controller_features.device,
            dtype=torch.int64,
        )
        return CandidateReadoutContext(
            query=self.query.index_select(resolved),
            source=self.source.index_select(resolved),
            edge=self.edge.index_select(resolved),
            destination=self.destination.index_select(resolved),
            global_evidence=self.global_evidence.index_select(resolved),
            controller_features=self.controller_features[resolved],
        )


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
    readout_context: CandidateReadoutContext | None = None

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
            )),
            readout_context=self.readout_context,
        )
