from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
import torch

from ..config import GraphSchema, PackConfig
from ..graph import PackedGraph, pack_graph_from_topology
from ..manifold import PackedManifoldFamily, pack_manifold_family
from ..topology import TopologyComponent, pack_topology
from .features import SREVocabulary, stable_u63
from .schema import SRE_ADVERSARY_FAMILIES, SRERetrievalCase


@dataclass(frozen=True, slots=True)
class PackedSRERetrievalBatch:
    graph: PackedGraph
    query: PackedManifoldFamily
    cases: tuple[SRERetrievalCase, ...]
    root_node_ids: torch.Tensor
    candidate_node_ids: torch.Tensor
    candidate_features: torch.Tensor
    relevance: torch.Tensor
    hard_negative: torch.Tensor
    adversary: torch.Tensor
    tie_break: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.graph.device

    @property
    def batch_size(self) -> int:
        return len(self.cases)

    @property
    def pool_size(self) -> int:
        return int(self.candidate_node_ids.shape[1])

    def validate_execution_path(self) -> None:
        expansion = self.graph.topology.expand_frontier(self.root_node_ids)
        if expansion.total_arcs != self.batch_size * self.pool_size:
            raise ValueError("query-root frontier did not enumerate every candidate")
        if not torch.equal(
            expansion.destination_node_ids.reshape_as(self.candidate_node_ids),
            self.candidate_node_ids,
        ):
            raise ValueError("packed frontier order does not align with candidates")


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _edge_value(
    *,
    width: int,
    edge_type: str | None,
    effective_at: str | None,
    request_time: str,
    vocabulary: SREVocabulary,
) -> torch.Tensor:
    value = torch.zeros((1, width), dtype=torch.float32)
    if edge_type is None:
        value[0, 0] = 1.0
        return value
    try:
        edge_id = vocabulary.edge_types.index(edge_type)
    except ValueError:
        edge_id = 0
    value[0, 1 + edge_id % max(width - 4, 1)] = 1.0
    delta_days = (
        _timestamp(request_time) - _timestamp(effective_at or request_time)
    ).total_seconds() / 86_400
    value[0, -3] = np.sign(delta_days) * np.log1p(abs(delta_days))
    value[0, -2] = float(delta_days >= 0)
    value[0, -1] = float(delta_days < 0)
    return value


def pack_sre_batch(
    cases: Sequence[SRERetrievalCase],
    *,
    query_embeddings: torch.Tensor,
    incoming_embeddings: torch.Tensor,
    incoming_present: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_features: torch.Tensor,
    vocabulary: SREVocabulary,
    device: torch.device | str = "cpu",
    validate: bool = True,
) -> PackedSRERetrievalBatch:
    if not cases:
        raise ValueError("an SRE packed batch requires cases")
    batch_size = len(cases)
    pool_size = cases[0].pool_size
    if any(case.pool_size != pool_size for case in cases):
        raise ValueError("all SRE pools in a batch must have equal size")
    if query_embeddings.ndim != 2:
        raise ValueError("query_embeddings must have shape [batch, width]")
    width = int(query_embeddings.shape[1])
    expected = (batch_size, width)
    if tuple(query_embeddings.shape) != expected or tuple(incoming_embeddings.shape) != expected:
        raise ValueError("query and incoming embeddings are not batch-aligned")
    if tuple(incoming_present.shape) != (batch_size,):
        raise ValueError("incoming_present must be a batch vector")
    if tuple(candidate_embeddings.shape) != (batch_size, pool_size, width):
        raise ValueError("candidate embeddings are not pool-aligned")
    if tuple(candidate_features.shape[:2]) != (batch_size, pool_size):
        raise ValueError("candidate features are not pool-aligned")
    schema = GraphSchema(
        summary_dim=width,
        context_dim=width,
        edge_dim=width,
    )
    components = []
    summaries: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    query_rows: list[torch.Tensor] = []
    for graph_index, case in enumerate(cases):
        candidate_index = {
            candidate.candidate_id: index + 1
            for index, candidate in enumerate(case.candidates)
        }
        edge_src = [0] * pool_size
        edge_dst = list(range(1, pool_size + 1))
        bidirectional = [False] * pool_size
        edge_values = [
            _edge_value(
                width=width,
                edge_type=None,
                effective_at=None,
                request_time=case.request_time,
                vocabulary=vocabulary,
            )
            for _ in range(pool_size)
        ]
        for relationship in case.relationships:
            edge_src.append(candidate_index[relationship.source_memory_id])
            edge_dst.append(candidate_index[relationship.destination_memory_id])
            bidirectional.append(True)
            edge_values.append(
                _edge_value(
                    width=width,
                    edge_type=relationship.edge_type,
                    effective_at=relationship.effective_at,
                    request_time=case.request_time,
                    vocabulary=vocabulary,
                )
            )
        components.append(
            TopologyComponent(
                node_count=pool_size + 1,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_bidirectional=bidirectional,
                schema=schema,
            )
        )
        summaries.append(query_embeddings[graph_index].reshape(1, width))
        summaries.extend(
            candidate_embeddings[graph_index, candidate_index].reshape(1, width)
            for candidate_index in range(pool_size)
        )
        contexts.extend(
            torch.empty((0, width), dtype=query_embeddings.dtype)
            for _ in range(pool_size + 1)
        )
        edges.extend(edge_values)
        rows = [query_embeddings[graph_index]]
        if bool(incoming_present[graph_index].item()):
            rows.append(incoming_embeddings[graph_index])
        query_rows.append(torch.stack(rows))
    target = torch.device(device)
    pack_config = PackConfig(device=target, value_dtype=torch.float32)
    topology = pack_topology(
        components,
        device=target,
        schema=schema,
        validate=validate,
    )
    graph = pack_graph_from_topology(
        topology,
        summaries,
        contexts,
        edges,
        pack_config=pack_config,
        schema=schema,
        validate=validate,
    )
    query = pack_manifold_family(
        query_rows,
        owner_count=batch_size,
        width=width,
        owner_graph_ids=torch.arange(batch_size, dtype=torch.int32, device=target),
        resolved_pack_config=graph.resolved_pack_config,
        allow_empty=False,
        family_name="sre_query",
        validate=validate,
    )
    graph_starts = topology.graph_node_ptr[:-1].to(torch.int64)
    root_ids = graph_starts.to(torch.int32)
    candidate_ids = (
        graph_starts[:, None]
        + torch.arange(1, pool_size + 1, device=target)[None, :]
    ).to(torch.int32)
    adversary_to_id = {
        name: index for index, name in enumerate(SRE_ADVERSARY_FAMILIES)
    }
    relevance = torch.tensor(
        [case.relevance for case in cases], dtype=torch.bool, device=target
    )
    adversary = torch.tensor(
        [
            [
                -1 if label.relevant else adversary_to_id[label.adversary or "mixed_adversarial"]
                for label in case.labels
            ]
            for case in cases
        ],
        dtype=torch.int16,
        device=target,
    )
    hard_negative = torch.tensor(
        [[label.hard_negative for label in case.labels] for case in cases],
        dtype=torch.bool,
        device=target,
    )
    tie_break = torch.tensor(
        [
            [stable_u63(candidate.candidate_id) for candidate in case.candidates]
            for case in cases
        ],
        dtype=torch.int64,
        device=target,
    )
    result = PackedSRERetrievalBatch(
        graph=graph,
        query=query,
        cases=tuple(cases),
        root_node_ids=root_ids,
        candidate_node_ids=candidate_ids,
        candidate_features=candidate_features.to(target, dtype=torch.float32),
        relevance=relevance,
        hard_negative=hard_negative,
        adversary=adversary,
        tie_break=tie_break,
    )
    if validate:
        result.validate_execution_path()
    return result
