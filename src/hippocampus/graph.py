from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ._validation import infer_common_device, infer_common_dtype, require_floating
from .config import (
    ExecutionMode,
    ExecutionPolicy,
    GraphSchema,
    PackConfig,
    ResolvedPackConfig,
    resolve_execution_policy,
    resolve_pack_config,
)
from .manifold import (
    DenseCandidateComponents,
    PackedManifoldFamily,
    RaggedManifoldComponents,
    is_ragged_tuple,
    normalise_manifold_components,
    pack_manifold_family,
    value_tensors,
)
from .topology import (
    FrontierExpansion,
    PackedTopology,
    TopologyComponent,
    pack_topology,
)


@dataclass(frozen=True, slots=True)
class GraphComponents:
    """One graph's topology and three manifold families before packing."""

    schema: GraphSchema
    summaries: Any
    contexts: Any
    edges: Any
    edge_src: Any = ()
    edge_dst: Any = ()
    edge_bidirectional: Any = False
    node_count: int | None = None

    def __post_init__(self) -> None:
        inferred = _infer_owner_count(self.summaries)
        if self.node_count is None:
            if inferred is None:
                raise ValueError(
                    "node_count is required when it cannot be inferred from summaries"
                )
            object.__setattr__(self, "node_count", inferred)
        if self.node_count is None or self.node_count < 0:
            raise ValueError("node_count must be non-negative")
        if inferred is not None and inferred != self.node_count:
            raise ValueError("summary owner count disagrees with node_count")

    @property
    def summary_manifolds(self) -> Any:
        return self.summaries

    @property
    def context_manifolds(self) -> Any:
        return self.contexts

    @property
    def edge_manifolds(self) -> Any:
        return self.edges

    @property
    def topology_component(self) -> TopologyComponent:
        assert self.node_count is not None
        return TopologyComponent(
            node_count=self.node_count,
            edge_src=self.edge_src,
            edge_dst=self.edge_dst,
            edge_bidirectional=self.edge_bidirectional,
            schema=self.schema,
        )


GraphComponent = GraphComponents


@dataclass(frozen=True, slots=True)
class PackedGraph:
    topology: PackedTopology
    summaries: PackedManifoldFamily
    contexts: PackedManifoldFamily
    edges: PackedManifoldFamily
    schema: GraphSchema
    resolved_pack_config: ResolvedPackConfig

    @classmethod
    def unsafe_from_packed(
        cls,
        *,
        topology: PackedTopology,
        summaries: PackedManifoldFamily,
        contexts: PackedManifoldFamily,
        edges: PackedManifoldFamily,
        schema: GraphSchema | None = None,
        resolved_pack_config: ResolvedPackConfig | None = None,
    ) -> "PackedGraph":
        """Build a deeply validated zero-copy graph snapshot.

        All supplied topology and manifold tensors remain externally aliased.
        """

        resolved_schema = schema or topology.schema
        if resolved_schema is None:
            raise ValueError("unsafe graph construction requires a GraphSchema")
        resolved = resolved_pack_config or summaries.resolved_pack_config
        graph = cls(
            topology=topology,
            summaries=summaries,
            contexts=contexts,
            edges=edges,
            schema=resolved_schema,
            resolved_pack_config=resolved,
        )
        graph.validate()
        return graph

    @property
    def summary(self) -> PackedManifoldFamily:
        return self.summaries

    @property
    def context(self) -> PackedManifoldFamily:
        return self.contexts

    @property
    def edge(self) -> PackedManifoldFamily:
        return self.edges

    @property
    def device(self) -> torch.device:
        return self.topology.device

    @property
    def dtype(self) -> torch.dtype:
        return self.resolved_pack_config.value_dtype

    @property
    def execution_policy(self) -> ExecutionPolicy:
        return self.resolved_pack_config.execution_policy

    def validate(self) -> "PackedGraph":
        self.topology.validate()
        for family in (self.summaries, self.contexts, self.edges):
            family.validate()
            if family.device != self.topology.device:
                raise ValueError("all manifold families must share topology device")
            if family.resolved_pack_config != self.resolved_pack_config:
                raise ValueError(
                    "all manifold families must share resolved pack configuration"
                )
        if self.topology.schema is not None and self.topology.schema != self.schema:
            raise ValueError("topology schema disagrees with packed graph schema")
        if self.summaries.owner_count != self.topology.node_count:
            raise ValueError("summary owners must align with topology nodes")
        if self.contexts.owner_count != self.topology.node_count:
            raise ValueError("context owners must align with topology nodes")
        if self.edges.owner_count != self.topology.edge_count:
            raise ValueError("edge manifold owners must align with logical edges")
        if self.summaries.width != self.schema.summary_dim:
            raise ValueError("summary width disagrees with schema")
        if self.contexts.width != self.schema.context_dim:
            raise ValueError("context width disagrees with schema")
        if self.edges.width != self.schema.edge_dim:
            raise ValueError("edge width disagrees with schema")
        if self.summaries.owner_count and torch.any(
            self.summaries.lengths == 0
        ).item():
            raise ValueError("summary manifolds may not be empty")
        if (
            not self.schema.allow_empty_edge_manifolds
            and self.edges.owner_count
            and torch.any(self.edges.lengths == 0).item()
        ):
            raise ValueError("empty edge manifolds require schema opt-in")
        if self.resolved_pack_config.device != self.topology.device:
            raise ValueError("resolved pack device disagrees with topology")
        return self

    def expand_frontier(
        self, frontier_node_ids: Any, *, validate_ids: bool = True
    ) -> FrontierExpansion:
        return self.topology.expand_frontier(
            frontier_node_ids, validate_ids=validate_ids
        )


def _infer_owner_count(values: Any) -> int | None:
    if isinstance(values, PackedManifoldFamily):
        return values.owner_count
    if isinstance(values, (RaggedManifoldComponents, DenseCandidateComponents)):
        return values.owner_count
    if isinstance(values, torch.Tensor) and values.ndim >= 1:
        return int(values.shape[0])
    if is_ragged_tuple(values):
        offsets = values[1]
        return (
            int(offsets.numel() - 1)
            if isinstance(offsets, torch.Tensor)
            else len(offsets) - 1
        )
    if isinstance(values, Sequence):
        return len(values)
    return None


def _resolve_schema(
    topology: PackedTopology,
    schema: GraphSchema | None,
) -> GraphSchema:
    if schema is not None:
        if topology.schema is not None and topology.schema != schema:
            raise ValueError("explicit schema disagrees with topology schema")
        return schema
    if topology.schema is not None:
        return topology.schema
    raise ValueError(
        "pack_graph_from_topology requires a GraphSchema supplied by the "
        "PackedTopology or the explicit schema keyword"
    )


def _resolve_value_configuration(
    families: Sequence[Any],
    *,
    pack_config: PackConfig | None,
    execution_policy: ExecutionPolicy | ExecutionMode | None,
    fallback_device: torch.device | str | None,
) -> ResolvedPackConfig:
    tensors = tuple(
        tensor for family in families for tensor in value_tensors(family)
    )
    for index, tensor in enumerate(tensors):
        require_floating(tensor, f"value tensor {index}")

    config = pack_config or PackConfig()
    source_device = (
        None
        if config.device is not None
        else infer_common_device(tensors, what="manifold value")
    )
    source_dtype = (
        None
        if config.value_dtype is not None
        else infer_common_dtype(tensors, what="manifold value")
    )
    return resolve_pack_config(
        pack_config,
        source_device=source_device,
        source_dtype=source_dtype,
        execution_policy=execution_policy,
        fallback_device=fallback_device,
        fallback_dtype=torch.float32,
    )


def pack_graph_from_topology(
    topology: PackedTopology,
    summaries: Any,
    contexts: Any,
    edges: Any,
    *,
    pack_config: PackConfig | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
    validate: bool = True,
    schema: GraphSchema | None = None,
) -> PackedGraph:
    """Create one differentiable value snapshot over reusable topology.

    The topology is never copied implicitly. Move it once with
    ``topology.to(device)`` before calling this function.
    """

    if not isinstance(topology, PackedTopology):
        raise TypeError("topology must be a PackedTopology")
    resolved_schema = _resolve_schema(topology, schema)
    resolved_policy = (
        topology.execution_policy
        if execution_policy is None
        else resolve_execution_policy(execution_policy)
    )
    resolved = _resolve_value_configuration(
        (summaries, contexts, edges),
        pack_config=pack_config,
        execution_policy=resolved_policy,
        fallback_device=topology.device,
    )
    if resolved.device != topology.device:
        raise ValueError(
            f"topology is on {topology.device}, but manifold values resolve to "
            f"{resolved.device}; call topology.to({resolved.device!s}) explicitly"
        )

    summary_family = pack_manifold_family(
        summaries,
        owner_count=topology.node_count,
        width=resolved_schema.summary_dim,
        owner_graph_ids=topology.node_graph_ids,
        resolved_pack_config=resolved,
        allow_empty=False,
        family_name="summaries",
        validate=validate,
    )
    context_family = pack_manifold_family(
        contexts,
        owner_count=topology.node_count,
        width=resolved_schema.context_dim,
        owner_graph_ids=topology.node_graph_ids,
        resolved_pack_config=resolved,
        allow_empty=True,
        family_name="contexts",
        validate=validate,
    )
    edge_family = pack_manifold_family(
        edges,
        owner_count=topology.edge_count,
        width=resolved_schema.edge_dim,
        owner_graph_ids=topology.edge_graph_ids,
        resolved_pack_config=resolved,
        allow_empty=resolved_schema.allow_empty_edge_manifolds,
        family_name="edges",
        validate=validate,
    )
    graph = PackedGraph(
        topology=topology,
        summaries=summary_family,
        contexts=context_family,
        edges=edge_family,
        schema=resolved_schema,
        resolved_pack_config=resolved,
    )
    if validate:
        graph.validate()
    return graph


def _mapping_value(mapping_or_object: Any, *names: str, default: Any = None) -> Any:
    if isinstance(mapping_or_object, Mapping):
        for name in names:
            if name in mapping_or_object:
                return mapping_or_object[name]
        return default
    for name in names:
        if hasattr(mapping_or_object, name):
            return getattr(mapping_or_object, name)
    return default


def _normalise_graph_component(
    component: Any, schema: GraphSchema | None
) -> GraphComponents:
    if isinstance(component, GraphComponents):
        if schema is not None and component.schema != schema:
            raise ValueError("component schema disagrees with batch schema")
        return component
    resolved_schema = _mapping_value(component, "schema", default=schema)
    if resolved_schema is None:
        raise ValueError("each graph component must provide a GraphSchema")
    return GraphComponents(
        schema=resolved_schema,
        summaries=_mapping_value(component, "summaries", "summary_manifolds"),
        contexts=_mapping_value(component, "contexts", "context_manifolds"),
        edges=_mapping_value(component, "edges", "edge_manifolds"),
        edge_src=_mapping_value(component, "edge_src", "src", default=()),
        edge_dst=_mapping_value(component, "edge_dst", "dst", default=()),
        edge_bidirectional=_mapping_value(
            component, "edge_bidirectional", "bidirectional", default=False
        ),
        node_count=_mapping_value(component, "node_count", "num_nodes"),
    )


def _concatenate_local_families(
    families: Sequence[PackedManifoldFamily],
    *,
    owner_graph_ids: torch.Tensor,
    resolved: ResolvedPackConfig,
    family_name: str,
) -> PackedManifoldFamily:
    if not families:
        raise ValueError("at least one local family is required")
    presence_modes = {family.presence is not None for family in families}
    if len(presence_modes) > 1:
        raise ValueError(
            f"{family_name} components must either all carry presence or all omit it"
        )
    candidate_modes = {
        family.candidate_slot_ids is not None for family in families
    }
    if len(candidate_modes) > 1:
        raise ValueError(
            f"{family_name} components may not mix dense-candidate provenance "
            "with provenance-free ragged values"
        )
    values = torch.cat([family.values for family in families], dim=0)
    lengths = torch.cat([family.lengths for family in families], dim=0)
    offsets64 = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=resolved.device),
            torch.cumsum(lengths.to(torch.int64), dim=0),
        )
    )
    offsets = offsets64.to(torch.int32)
    row_owner_ids = torch.repeat_interleave(
        torch.arange(
            lengths.numel(), dtype=torch.int32, device=resolved.device
        ),
        lengths.to(torch.int64),
    )
    presence = (
        torch.cat([family.presence for family in families], dim=0)
        if True in presence_modes
        else None
    )
    candidate_slot_ids = (
        torch.cat(
            [family.candidate_slot_ids for family in families], dim=0
        )
        if True in candidate_modes
        else None
    )
    return PackedManifoldFamily.unsafe_from_packed(
        values=values,
        offsets=offsets,
        row_owner_ids=row_owner_ids,
        owner_graph_ids=owner_graph_ids,
        lengths=lengths,
        presence=presence,
        candidate_slot_ids=candidate_slot_ids,
        resolved_pack_config=resolved,
    )


def pack_graph_batch_from_components(
    components: Sequence[Any],
    schema: GraphSchema | None = None,
    *,
    pack_config: PackConfig | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
    validate: bool = True,
) -> PackedGraph:
    """One-off convenience wrapper around topology and value snapshot packing."""

    normalised = tuple(
        _normalise_graph_component(component, schema) for component in components
    )
    schemas = {component.schema for component in normalised}
    if schema is not None:
        schemas.add(schema)
    if not schemas:
        if schema is None:
            raise ValueError("an empty graph batch requires an explicit GraphSchema")
        schemas.add(schema)
    if len(schemas) != 1:
        raise ValueError("all graph components must share one GraphSchema")
    resolved_schema = next(iter(schemas))

    all_family_inputs = tuple(
        family
        for component in normalised
        for family in (component.summaries, component.contexts, component.edges)
    )
    resolved = _resolve_value_configuration(
        all_family_inputs,
        pack_config=pack_config,
        execution_policy=execution_policy,
        fallback_device="cpu",
    )
    explicit_resolved_config = PackConfig(
        device=resolved.device,
        value_dtype=resolved.value_dtype,
        index_policy=resolved.index_policy,
        pin_cpu_staging=resolved.pin_cpu_staging,
    )
    topology = pack_topology(
        [component.topology_component for component in normalised],
        device=resolved.device,
        execution_policy=resolved.execution_policy,
        validate=validate,
        schema=resolved_schema,
    )

    if not normalised:
        empty_summary = RaggedManifoldComponents(
            torch.empty(
                (0, resolved_schema.summary_dim),
                device=resolved.device,
                dtype=resolved.value_dtype,
            ),
            torch.zeros(1, dtype=torch.int32, device=resolved.device),
        )
        empty_context = RaggedManifoldComponents(
            torch.empty(
                (0, resolved_schema.context_dim),
                device=resolved.device,
                dtype=resolved.value_dtype,
            ),
            torch.zeros(1, dtype=torch.int32, device=resolved.device),
        )
        empty_edge = RaggedManifoldComponents(
            torch.empty(
                (0, resolved_schema.edge_dim),
                device=resolved.device,
                dtype=resolved.value_dtype,
            ),
            torch.zeros(1, dtype=torch.int32, device=resolved.device),
        )
        return pack_graph_from_topology(
            topology,
            empty_summary,
            empty_context,
            empty_edge,
            pack_config=explicit_resolved_config,
            execution_policy=resolved.execution_policy,
            validate=validate,
            schema=resolved_schema,
        )

    local_summaries: list[PackedManifoldFamily] = []
    local_contexts: list[PackedManifoldFamily] = []
    local_edges: list[PackedManifoldFamily] = []
    for graph_id, component in enumerate(normalised):
        node_start = int(topology.graph_node_ptr[graph_id].item())
        node_end = int(topology.graph_node_ptr[graph_id + 1].item())
        edge_start = int(topology.graph_edge_ptr[graph_id].item())
        edge_end = int(topology.graph_edge_ptr[graph_id + 1].item())
        local_summaries.append(
            pack_manifold_family(
                component.summaries,
                owner_count=node_end - node_start,
                width=resolved_schema.summary_dim,
                owner_graph_ids=topology.node_graph_ids[node_start:node_end],
                resolved_pack_config=resolved,
                allow_empty=False,
                family_name=f"components[{graph_id}].summaries",
                validate=validate,
            )
        )
        local_contexts.append(
            pack_manifold_family(
                component.contexts,
                owner_count=node_end - node_start,
                width=resolved_schema.context_dim,
                owner_graph_ids=topology.node_graph_ids[node_start:node_end],
                resolved_pack_config=resolved,
                allow_empty=True,
                family_name=f"components[{graph_id}].contexts",
                validate=validate,
            )
        )
        local_edges.append(
            pack_manifold_family(
                component.edges,
                owner_count=edge_end - edge_start,
                width=resolved_schema.edge_dim,
                owner_graph_ids=topology.edge_graph_ids[edge_start:edge_end],
                resolved_pack_config=resolved,
                allow_empty=resolved_schema.allow_empty_edge_manifolds,
                family_name=f"components[{graph_id}].edges",
                validate=validate,
            )
        )

    summaries = _concatenate_local_families(
        local_summaries,
        owner_graph_ids=topology.node_graph_ids,
        resolved=resolved,
        family_name="summary",
    )
    contexts = _concatenate_local_families(
        local_contexts,
        owner_graph_ids=topology.node_graph_ids,
        resolved=resolved,
        family_name="context",
    )
    edge_values = _concatenate_local_families(
        local_edges,
        owner_graph_ids=topology.edge_graph_ids,
        resolved=resolved,
        family_name="edge",
    )
    return pack_graph_from_topology(
        topology,
        summaries,
        contexts,
        edge_values,
        pack_config=explicit_resolved_config,
        execution_policy=resolved.execution_policy,
        validate=validate,
        schema=resolved_schema,
    )
