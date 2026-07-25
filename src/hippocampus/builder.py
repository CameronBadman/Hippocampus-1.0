from __future__ import annotations

from typing import Any

import torch

from .config import (
    ExecutionMode,
    ExecutionPolicy,
    GraphSchema,
    PackConfig,
)
from .graph import GraphComponents, PackedGraph, pack_graph_batch_from_components


def _as_rows(value: Any, *, width: int, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape [rows, {width}]")
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    return tensor


class GraphBuilder:
    """Ergonomic builder for static fixtures and inference-time graphs."""

    def __init__(self, schema: GraphSchema) -> None:
        if not isinstance(schema, GraphSchema):
            raise TypeError("schema must be a GraphSchema")
        self.schema = schema
        self._summaries: list[torch.Tensor] = []
        self._contexts: list[torch.Tensor] = []
        self._edge_src: list[int] = []
        self._edge_dst: list[int] = []
        self._edge_bidirectional: list[bool] = []
        self._edges: list[torch.Tensor] = []

    @property
    def node_count(self) -> int:
        return len(self._summaries)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_node(self, summary: Any, context: Any | None = None) -> int:
        summary_rows = _as_rows(
            summary, width=self.schema.summary_dim, name="summary"
        )
        if summary_rows.shape[0] == 0:
            raise ValueError("summary manifolds may not be empty")
        if context is None:
            context_rows = torch.empty(
                (0, self.schema.context_dim),
                dtype=summary_rows.dtype,
                device=summary_rows.device,
            )
        else:
            context_rows = _as_rows(
                context, width=self.schema.context_dim, name="context"
            )
        node_id = self.node_count
        self._summaries.append(summary_rows)
        self._contexts.append(context_rows)
        return node_id

    def add_edge(
        self,
        source_node_id: int,
        destination_node_id: int,
        manifold: Any,
        *,
        bidirectional: bool = False,
    ) -> int:
        for name, node_id in (
            ("source_node_id", source_node_id),
            ("destination_node_id", destination_node_id),
        ):
            if isinstance(node_id, bool) or not isinstance(node_id, int):
                raise TypeError(f"{name} must be an integer")
            if node_id < 0 or node_id >= self.node_count:
                raise IndexError(
                    f"{name}={node_id} is outside [0, {self.node_count})"
                )
        edge_rows = _as_rows(
            manifold, width=self.schema.edge_dim, name="edge manifold"
        )
        if (
            edge_rows.shape[0] == 0
            and not self.schema.allow_empty_edge_manifolds
        ):
            raise ValueError("empty edge manifolds require schema opt-in")
        edge_id = self.edge_count
        self._edge_src.append(source_node_id)
        self._edge_dst.append(destination_node_id)
        self._edge_bidirectional.append(bool(bidirectional))
        self._edges.append(edge_rows)
        return edge_id

    add_logical_edge = add_edge

    def as_components(self) -> GraphComponents:
        return GraphComponents(
            schema=self.schema,
            summaries=tuple(self._summaries),
            contexts=tuple(self._contexts),
            edges=tuple(self._edges),
            edge_src=tuple(self._edge_src),
            edge_dst=tuple(self._edge_dst),
            edge_bidirectional=tuple(self._edge_bidirectional),
            node_count=self.node_count,
        )

    @property
    def topology_component(self):
        return self.as_components().topology_component

    def compile(
        self,
        pack_config: PackConfig | None = None,
        *,
        execution_policy: ExecutionPolicy | ExecutionMode | None = None,
        validate: bool = True,
    ) -> PackedGraph:
        # Static builder defaults are intentionally CPU FP32, independent of
        # the input tensors' current placement or precision.
        effective_config = pack_config or PackConfig(
            device="cpu", value_dtype=torch.float32
        )
        return pack_graph_batch_from_components(
            [self.as_components()],
            self.schema,
            pack_config=effective_config,
            execution_policy=execution_policy,
            validate=validate,
        )

