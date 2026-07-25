from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class HypothesisBatch:
    node_ids: torch.Tensor
    graph_ids: torch.Tensor
    path_state: torch.Tensor
    scores: torch.Tensor
    depths: torch.Tensor
    parent_trace_ids: torch.Tensor
    incoming_arc_ids: torch.Tensor
    incoming_edge_ids: torch.Tensor
    context_read: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.node_ids.numel())

    @property
    def device(self) -> torch.device:
        return self.path_state.device

    def validate(self) -> "HypothesisBatch":
        count = self.count
        if self.path_state.ndim != 3 or self.path_state.shape[0] != count:
            raise ValueError("path_state must have shape [hypotheses, rows, width]")
        expected = {
            "node_ids": torch.int32,
            "graph_ids": torch.int32,
            "depths": torch.int32,
            "parent_trace_ids": torch.int64,
            "incoming_arc_ids": torch.int32,
            "incoming_edge_ids": torch.int32,
            "context_read": torch.bool,
        }
        for name, dtype in expected.items():
            tensor = getattr(self, name)
            if tensor.ndim != 1 or tensor.numel() != count:
                raise ValueError(f"{name} must align with hypotheses")
            if tensor.dtype != dtype:
                raise TypeError(f"{name} must use {dtype}")
            if tensor.device != self.device:
                raise ValueError(f"{name} must share path_state device")
        if (
            self.scores.ndim != 1
            or self.scores.numel() != count
            or self.scores.device != self.device
        ):
            raise ValueError("scores must align with hypotheses")
        return self

    def repeat_occurrences(self, positions: torch.Tensor) -> "HypothesisBatch":
        index = positions.to(device=self.device, dtype=torch.int64)
        return HypothesisBatch(
            node_ids=self.node_ids[index],
            graph_ids=self.graph_ids[index],
            path_state=self.path_state[index],
            scores=self.scores[index],
            depths=self.depths[index],
            parent_trace_ids=self.parent_trace_ids[index],
            incoming_arc_ids=self.incoming_arc_ids[index],
            incoming_edge_ids=self.incoming_edge_ids[index],
            context_read=self.context_read[index],
        )
