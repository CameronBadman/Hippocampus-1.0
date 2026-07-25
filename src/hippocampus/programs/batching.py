from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..config import GraphSchema, PackConfig
from ..graph import PackedGraph, pack_graph_from_topology
from ..manifold import PackedManifoldFamily, pack_manifold_family
from ..topology import TopologyComponent, pack_topology
from .renderer import RenderedCase
from .schema import GraphProgramCase


@dataclass(frozen=True, slots=True)
class PackedProgramBatch:
    graph: PackedGraph
    query: PackedManifoldFamily
    cases: tuple[GraphProgramCase, ...]

    @property
    def device(self) -> torch.device:
        return self.graph.device

    @property
    def graph_count(self) -> int:
        return len(self.cases)

    def global_node_id(self, graph_id: int, local_node_id: int) -> int:
        start = int(self.graph.topology.graph_node_ptr[graph_id].item())
        return start + local_node_id

    def global_edge_id(self, graph_id: int, local_edge_id: int) -> int:
        start = int(self.graph.topology.graph_edge_ptr[graph_id].item())
        return start + local_edge_id


def pack_rendered_cases(
    cases: Sequence[GraphProgramCase],
    rendered: Sequence[RenderedCase],
    *,
    schema: GraphSchema,
    pack_config: PackConfig | None = None,
    validate: bool = True,
) -> PackedProgramBatch:
    if len(cases) != len(rendered):
        raise ValueError("cases and rendered values must have equal lengths")
    if not cases:
        raise ValueError("a packed program batch requires at least one case")
    for index, (case, values) in enumerate(zip(cases, rendered, strict=True)):
        if case.case_id != values.case_id:
            raise ValueError(f"rendered case {index} does not match supervisor case")
        if len(values.summaries) != len(case.nodes):
            raise ValueError(f"rendered case {index} summaries do not align")
        if len(values.contexts) != len(case.nodes):
            raise ValueError(f"rendered case {index} contexts do not align")
        if len(values.edges) != len(case.edges):
            raise ValueError(f"rendered case {index} edges do not align")

    topology = pack_topology(
        [
            TopologyComponent(
                node_count=len(case.nodes),
                edge_src=[edge.source_node for edge in case.edges],
                edge_dst=[edge.destination_node for edge in case.edges],
                edge_bidirectional=[edge.bidirectional for edge in case.edges],
                schema=schema,
            )
            for case in cases
        ],
        device=(
            pack_config.device
            if pack_config is not None and pack_config.device is not None
            else rendered[0].query.device
        ),
        schema=schema,
        validate=validate,
    )
    summaries = tuple(
        rows for values in rendered for rows in values.summaries
    )
    contexts = tuple(
        rows for values in rendered for rows in values.contexts
    )
    edges = tuple(rows for values in rendered for rows in values.edges)
    graph = pack_graph_from_topology(
        topology,
        summaries,
        contexts,
        edges,
        pack_config=pack_config,
        schema=schema,
        validate=validate,
    )

    query_dim = rendered[0].query_dim
    if any(values.query_dim != query_dim for values in rendered):
        raise ValueError("all rendered query manifolds must share one width")
    query = pack_manifold_family(
        tuple(values.query for values in rendered),
        owner_count=len(cases),
        width=query_dim,
        owner_graph_ids=torch.arange(
            len(cases),
            dtype=torch.int32,
            device=graph.device,
        ),
        resolved_pack_config=graph.resolved_pack_config,
        allow_empty=False,
        family_name="query",
        validate=validate,
    )
    return PackedProgramBatch(
        graph=graph,
        query=query,
        cases=tuple(cases),
    )
