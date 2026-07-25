from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ._validation import (
    INT32_MAX,
    as_id_tensor,
    infer_common_device,
    require_int32_capacity,
    require_tensor,
    tensor_scalar_bool,
    validate_pointer,
)
from .config import (
    ExecutionMode,
    ExecutionPolicy,
    GraphSchema,
    resolve_execution_policy,
)


@dataclass(frozen=True, slots=True)
class TopologyComponent:
    """One graph's local logical-edge topology.

    Endpoints are local node IDs. A bidirectional logical edge contributes two
    traversal arcs while retaining one edge ID and one edge manifold.
    """

    node_count: int
    edge_src: Any = ()
    edge_dst: Any = ()
    edge_bidirectional: Any = False
    schema: GraphSchema | None = None

    def __post_init__(self) -> None:
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int):
            raise TypeError("node_count must be an integer")
        if self.node_count < 0:
            raise ValueError("node_count must be non-negative")

    @property
    def num_nodes(self) -> int:
        return self.node_count


GraphTopologyComponent = TopologyComponent


@dataclass(frozen=True, slots=True)
class FrontierExpansion:
    arc_ids: torch.Tensor
    edge_ids: torch.Tensor
    source_node_ids: torch.Tensor
    destination_node_ids: torch.Tensor
    frontier_positions: torch.Tensor
    arc_offsets: torch.Tensor

    @property
    def src_node_ids(self) -> torch.Tensor:
        return self.source_node_ids

    @property
    def dst_node_ids(self) -> torch.Tensor:
        return self.destination_node_ids

    @property
    def offsets(self) -> torch.Tensor:
        return self.arc_offsets

    @property
    def total_arcs(self) -> int:
        return int(self.arc_ids.numel())


def _all_same_device(tensors: Sequence[torch.Tensor], name: str) -> torch.device:
    device = tensors[0].device
    for tensor in tensors[1:]:
        if tensor.device != device:
            raise ValueError(f"all {name} tensors must be on {device}")
    return device


@dataclass(frozen=True, slots=True)
class PackedTopology:
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_bidirectional: torch.Tensor
    adjacency_row_ptr: torch.Tensor
    adjacency_dst: torch.Tensor
    adjacency_edge_id: torch.Tensor
    graph_node_ptr: torch.Tensor
    graph_edge_ptr: torch.Tensor
    graph_arc_ptr: torch.Tensor
    node_graph_ids: torch.Tensor
    edge_graph_ids: torch.Tensor
    execution_policy: ExecutionPolicy
    schema: GraphSchema | None = None

    @classmethod
    def unsafe_from_packed(
        cls,
        *,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_bidirectional: torch.Tensor,
        adjacency_row_ptr: torch.Tensor,
        adjacency_dst: torch.Tensor,
        adjacency_edge_id: torch.Tensor,
        graph_node_ptr: torch.Tensor,
        graph_edge_ptr: torch.Tensor,
        graph_arc_ptr: torch.Tensor,
        node_graph_ids: torch.Tensor,
        edge_graph_ids: torch.Tensor,
        execution_policy: ExecutionPolicy | ExecutionMode | None = None,
        schema: GraphSchema | None = None,
    ) -> "PackedTopology":
        """Build a deeply validated zero-copy topology.

        Every tensor is aliased exactly. External mutation therefore changes
        this snapshot and may invalidate its invariants.
        """

        topology = cls(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_bidirectional=edge_bidirectional,
            adjacency_row_ptr=adjacency_row_ptr,
            adjacency_dst=adjacency_dst,
            adjacency_edge_id=adjacency_edge_id,
            graph_node_ptr=graph_node_ptr,
            graph_edge_ptr=graph_edge_ptr,
            graph_arc_ptr=graph_arc_ptr,
            node_graph_ids=node_graph_ids,
            edge_graph_ids=edge_graph_ids,
            execution_policy=resolve_execution_policy(execution_policy),
            schema=schema,
        )
        topology.validate()
        return topology

    @property
    def device(self) -> torch.device:
        return self.edge_src.device

    @property
    def node_count(self) -> int:
        return int(self.node_graph_ids.numel())

    @property
    def num_nodes(self) -> int:
        return self.node_count

    @property
    def edge_count(self) -> int:
        return int(self.edge_src.numel())

    @property
    def num_edges(self) -> int:
        return self.edge_count

    @property
    def arc_count(self) -> int:
        return int(self.adjacency_dst.numel())

    @property
    def num_arcs(self) -> int:
        return self.arc_count

    @property
    def graph_count(self) -> int:
        return int(self.graph_node_ptr.numel() - 1)

    @property
    def num_graphs(self) -> int:
        return self.graph_count

    def validate(self) -> "PackedTopology":
        tensors = (
            self.edge_src,
            self.edge_dst,
            self.edge_bidirectional,
            self.adjacency_row_ptr,
            self.adjacency_dst,
            self.adjacency_edge_id,
            self.graph_node_ptr,
            self.graph_edge_ptr,
            self.graph_arc_ptr,
            self.node_graph_ids,
            self.edge_graph_ids,
        )
        device = _all_same_device(tensors, "packed topology")
        del device
        self.execution_policy.validate_global_state()

        for name in (
            "edge_src",
            "edge_dst",
            "adjacency_row_ptr",
            "adjacency_dst",
            "adjacency_edge_id",
            "graph_node_ptr",
            "graph_edge_ptr",
            "graph_arc_ptr",
            "node_graph_ids",
            "edge_graph_ids",
        ):
            require_tensor(
                getattr(self, name),
                name,
                ndim=1,
                dtype=torch.int32,
                contiguous=True,
            )
        require_tensor(
            self.edge_bidirectional,
            "edge_bidirectional",
            ndim=1,
            dtype=torch.bool,
            contiguous=True,
        )

        node_count = self.node_count
        edge_count = self.edge_count
        arc_count = self.arc_count
        graph_count = self.graph_count
        require_int32_capacity("node_count", node_count)
        require_int32_capacity("edge_count", edge_count)
        require_int32_capacity("arc_count", arc_count)

        if self.edge_dst.numel() != edge_count:
            raise ValueError("edge_src and edge_dst must have equal lengths")
        if self.edge_bidirectional.numel() != edge_count:
            raise ValueError("edge_bidirectional must align with logical edges")
        if self.edge_graph_ids.numel() != edge_count:
            raise ValueError("edge_graph_ids must align with logical edges")
        if self.node_graph_ids.numel() != node_count:
            raise ValueError("node_graph_ids must align with nodes")
        if self.adjacency_edge_id.numel() != arc_count:
            raise ValueError("adjacency_edge_id must align with adjacency_dst")
        if self.adjacency_row_ptr.numel() != node_count + 1:
            raise ValueError("adjacency_row_ptr must have node_count + 1 entries")
        if self.graph_edge_ptr.numel() != graph_count + 1:
            raise ValueError("graph_edge_ptr must align with graph_node_ptr")
        if self.graph_arc_ptr.numel() != graph_count + 1:
            raise ValueError("graph_arc_ptr must align with graph_node_ptr")

        validate_pointer(
            self.adjacency_row_ptr,
            "adjacency_row_ptr",
            expected_final=arc_count,
        )
        validate_pointer(
            self.graph_node_ptr, "graph_node_ptr", expected_final=node_count
        )
        validate_pointer(
            self.graph_edge_ptr, "graph_edge_ptr", expected_final=edge_count
        )
        validate_pointer(
            self.graph_arc_ptr, "graph_arc_ptr", expected_final=arc_count
        )

        if edge_count:
            if tensor_scalar_bool(torch.any(self.edge_src < 0)) or tensor_scalar_bool(
                torch.any(self.edge_src >= node_count)
            ):
                raise ValueError("edge_src contains an out-of-range node ID")
            if tensor_scalar_bool(torch.any(self.edge_dst < 0)) or tensor_scalar_bool(
                torch.any(self.edge_dst >= node_count)
            ):
                raise ValueError("edge_dst contains an out-of-range node ID")
        if arc_count:
            if tensor_scalar_bool(torch.any(self.adjacency_dst < 0)) or tensor_scalar_bool(
                torch.any(self.adjacency_dst >= node_count)
            ):
                raise ValueError("adjacency_dst contains an out-of-range node ID")
            if tensor_scalar_bool(
                torch.any(self.adjacency_edge_id < 0)
            ) or tensor_scalar_bool(torch.any(self.adjacency_edge_id >= edge_count)):
                raise ValueError("adjacency_edge_id contains an out-of-range edge ID")

        node_lengths = (self.graph_node_ptr[1:] - self.graph_node_ptr[:-1]).to(
            torch.int64
        )
        edge_lengths = (self.graph_edge_ptr[1:] - self.graph_edge_ptr[:-1]).to(
            torch.int64
        )
        graph_ids = torch.arange(
            graph_count, dtype=torch.int32, device=self.device
        )
        expected_node_graph_ids = torch.repeat_interleave(graph_ids, node_lengths)
        expected_edge_graph_ids = torch.repeat_interleave(graph_ids, edge_lengths)
        if not torch.equal(self.node_graph_ids, expected_node_graph_ids):
            raise ValueError("node_graph_ids disagree with graph_node_ptr")
        if not torch.equal(self.edge_graph_ids, expected_edge_graph_ids):
            raise ValueError("edge_graph_ids disagree with graph_edge_ptr")

        expected_graph_arc_ptr = self.adjacency_row_ptr[
            self.graph_node_ptr.to(torch.int64)
        ]
        if not torch.equal(self.graph_arc_ptr, expected_graph_arc_ptr):
            raise ValueError("graph_arc_ptr disagrees with graph node CSR boundaries")

        expected_arc_count = edge_count + int(self.edge_bidirectional.sum().item())
        if arc_count != expected_arc_count:
            raise ValueError(
                "arc count must equal edge_count + bidirectional_edge_count"
            )

        if edge_count:
            src_graph = self.node_graph_ids[self.edge_src.to(torch.int64)]
            dst_graph = self.node_graph_ids[self.edge_dst.to(torch.int64)]
            if not torch.equal(src_graph, self.edge_graph_ids) or not torch.equal(
                dst_graph, self.edge_graph_ids
            ):
                raise ValueError("logical edges may not cross packed graph boundaries")

        if arc_count:
            arc_lengths = (
                self.adjacency_row_ptr[1:] - self.adjacency_row_ptr[:-1]
            ).to(torch.int64)
            sources = torch.repeat_interleave(
                torch.arange(node_count, dtype=torch.int32, device=self.device),
                arc_lengths,
            )
            edge_ids = self.adjacency_edge_id.to(torch.int64)
            logical_src = self.edge_src[edge_ids]
            logical_dst = self.edge_dst[edge_ids]
            bidirectional = self.edge_bidirectional[edge_ids]
            forward = (sources == logical_src) & (
                self.adjacency_dst == logical_dst
            )
            reverse = (
                bidirectional
                & (sources == logical_dst)
                & (self.adjacency_dst == logical_src)
            )
            if tensor_scalar_bool(torch.any(~(forward | reverse))):
                raise ValueError("CSR arcs disagree with logical edge endpoints")
            if arc_count > 1:
                same_source = sources[1:] == sources[:-1]
                decreasing_edge = (
                    self.adjacency_edge_id[1:] < self.adjacency_edge_id[:-1]
                )
                if tensor_scalar_bool(torch.any(same_source & decreasing_edge)):
                    raise ValueError(
                        "CSR rows must retain canonical logical-edge ordering"
                    )
        return self

    def to(
        self, device: torch.device | str, non_blocking: bool = False
    ) -> "PackedTopology":
        target = torch.device(device)
        if target == self.device:
            return self

        def moved(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(target, non_blocking=non_blocking)

        return PackedTopology.unsafe_from_packed(
            edge_src=moved(self.edge_src),
            edge_dst=moved(self.edge_dst),
            edge_bidirectional=moved(self.edge_bidirectional),
            adjacency_row_ptr=moved(self.adjacency_row_ptr),
            adjacency_dst=moved(self.adjacency_dst),
            adjacency_edge_id=moved(self.adjacency_edge_id),
            graph_node_ptr=moved(self.graph_node_ptr),
            graph_edge_ptr=moved(self.graph_edge_ptr),
            graph_arc_ptr=moved(self.graph_arc_ptr),
            node_graph_ids=moved(self.node_graph_ids),
            edge_graph_ids=moved(self.edge_graph_ids),
            execution_policy=self.execution_policy,
            schema=self.schema,
        )

    def pin_memory(self) -> "PackedTopology":
        if self.device.type != "cpu":
            raise ValueError("only CPU topology snapshots can be pinned")
        if not torch.cuda.is_available():
            raise RuntimeError("pinning requires an available CUDA runtime")

        def pinned(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if tensor.is_pinned() else tensor.pin_memory()

        return PackedTopology.unsafe_from_packed(
            edge_src=pinned(self.edge_src),
            edge_dst=pinned(self.edge_dst),
            edge_bidirectional=pinned(self.edge_bidirectional),
            adjacency_row_ptr=pinned(self.adjacency_row_ptr),
            adjacency_dst=pinned(self.adjacency_dst),
            adjacency_edge_id=pinned(self.adjacency_edge_id),
            graph_node_ptr=pinned(self.graph_node_ptr),
            graph_edge_ptr=pinned(self.graph_edge_ptr),
            graph_arc_ptr=pinned(self.graph_arc_ptr),
            node_graph_ids=pinned(self.node_graph_ids),
            edge_graph_ids=pinned(self.edge_graph_ids),
            execution_policy=self.execution_policy,
            schema=self.schema,
        )

    def expand_frontier(
        self,
        frontier_node_ids: Any,
        *,
        validate_ids: bool = True,
    ) -> FrontierExpansion:
        frontier = as_id_tensor(
            frontier_node_ids,
            name="frontier_node_ids",
            device=self.device,
            validate_ids=validate_ids,
            upper_bound=self.node_count,
        )
        frontier_long = frontier.to(torch.int64)
        if frontier.numel() == 0:
            empty_i32 = torch.empty(0, dtype=torch.int32, device=self.device)
            empty_i64 = torch.empty(0, dtype=torch.int64, device=self.device)
            return FrontierExpansion(
                arc_ids=empty_i32,
                edge_ids=empty_i32.clone(),
                source_node_ids=empty_i32.clone(),
                destination_node_ids=empty_i32.clone(),
                frontier_positions=empty_i64,
                arc_offsets=torch.zeros(1, dtype=torch.int32, device=self.device),
            )

        starts = self.adjacency_row_ptr[frontier_long].to(torch.int64)
        ends = self.adjacency_row_ptr[frontier_long + 1].to(torch.int64)
        lengths64 = ends - starts
        total_arcs = int(lengths64.sum().item())
        require_int32_capacity("expanded frontier arc count", total_arcs)

        arc_offsets64 = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=self.device),
                torch.cumsum(lengths64, dim=0),
            )
        )
        positions = torch.repeat_interleave(
            torch.arange(frontier.numel(), dtype=torch.int64, device=self.device),
            lengths64,
        )
        if total_arcs:
            local_rows = (
                torch.arange(total_arcs, dtype=torch.int64, device=self.device)
                - arc_offsets64[positions]
            )
            arc_ids64 = starts[positions] + local_rows
            arc_ids = arc_ids64.to(torch.int32)
            edge_ids = self.adjacency_edge_id[arc_ids64]
            destinations = self.adjacency_dst[arc_ids64]
            sources = frontier.to(torch.int32)[positions]
        else:
            arc_ids = torch.empty(0, dtype=torch.int32, device=self.device)
            edge_ids = torch.empty(0, dtype=torch.int32, device=self.device)
            destinations = torch.empty(0, dtype=torch.int32, device=self.device)
            sources = torch.empty(0, dtype=torch.int32, device=self.device)

        return FrontierExpansion(
            arc_ids=arc_ids,
            edge_ids=edge_ids,
            source_node_ids=sources,
            destination_node_ids=destinations,
            frontier_positions=positions,
            arc_offsets=arc_offsets64.to(torch.int32),
        )


def _component_value(component: Any, *names: str, default: Any = None) -> Any:
    if isinstance(component, Mapping):
        for name in names:
            if name in component:
                return component[name]
        return default
    for name in names:
        if hasattr(component, name):
            return getattr(component, name)
    return default


def _normalise_component(component: Any) -> TopologyComponent:
    if isinstance(component, TopologyComponent):
        return component
    if hasattr(component, "topology_component"):
        value = component.topology_component
        return value() if callable(value) else value

    node_count = _component_value(component, "node_count", "num_nodes")
    if node_count is None:
        raise TypeError("topology components must define node_count or num_nodes")
    edge_index = _component_value(component, "edge_index")
    if edge_index is not None:
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2:
            raise ValueError("edge_index must be a rank-2 tensor")
        if edge_index.shape[0] == 2:
            edge_src, edge_dst = edge_index[0], edge_index[1]
        elif edge_index.shape[1] == 2:
            edge_src, edge_dst = edge_index[:, 0], edge_index[:, 1]
        else:
            raise ValueError("edge_index must have shape [2, E] or [E, 2]")
    else:
        edge_src = _component_value(component, "edge_src", "src", default=())
        edge_dst = _component_value(component, "edge_dst", "dst", default=())
    return TopologyComponent(
        node_count=int(node_count),
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_bidirectional=_component_value(
            component,
            "edge_bidirectional",
            "bidirectional",
            default=False,
        ),
        schema=_component_value(component, "schema"),
    )


def _structural_tensor(
    value: Any,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be rank 1")
        if dtype == torch.int64 and value.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"{name} must use an integer dtype")
        if dtype == torch.bool and value.dtype != torch.bool:
            raise TypeError(f"{name} must use bool dtype")
        return value.to(device=device, dtype=dtype).contiguous()
    return torch.as_tensor(value, device=device, dtype=dtype).contiguous()


def pack_topology(
    components: Sequence[Any],
    *,
    device: torch.device | str | None = None,
    execution_policy: ExecutionPolicy | ExecutionMode | None = None,
    validate: bool = True,
    schema: GraphSchema | None = None,
) -> PackedTopology:
    normalised = tuple(_normalise_component(component) for component in components)
    policy = resolve_execution_policy(execution_policy)

    component_schemas = {
        component.schema for component in normalised if component.schema is not None
    }
    if schema is not None:
        component_schemas.add(schema)
    if len(component_schemas) > 1:
        raise ValueError("all topology components must use one GraphSchema")
    resolved_schema = next(iter(component_schemas), None)

    structural_sources: list[torch.Tensor] = []
    for component in normalised:
        if isinstance(component.edge_src, torch.Tensor):
            structural_sources.append(component.edge_src)
        if isinstance(component.edge_dst, torch.Tensor):
            structural_sources.append(component.edge_dst)
    if device is None:
        source_device = infer_common_device(
            structural_sources, what="topology endpoint"
        )
        target = source_device or torch.device("cpu")
    else:
        target = torch.device(device)

    node_counts: list[int] = []
    edge_counts: list[int] = []
    local_src_parts: list[torch.Tensor] = []
    local_dst_parts: list[torch.Tensor] = []
    bidirectional_parts: list[torch.Tensor] = []
    total_nodes = 0
    total_edges = 0
    total_arcs = 0

    for graph_id, component in enumerate(normalised):
        node_count = require_int32_capacity(
            f"components[{graph_id}].node_count", component.node_count
        )
        src = _structural_tensor(
            component.edge_src,
            name=f"components[{graph_id}].edge_src",
            device=target,
            dtype=torch.int64,
        )
        dst = _structural_tensor(
            component.edge_dst,
            name=f"components[{graph_id}].edge_dst",
            device=target,
            dtype=torch.int64,
        )
        if src.numel() != dst.numel():
            raise ValueError(
                f"components[{graph_id}] edge_src and edge_dst lengths differ"
            )
        edge_count = require_int32_capacity(
            f"components[{graph_id}].edge_count", src.numel()
        )
        raw_bidirectional = component.edge_bidirectional
        if isinstance(raw_bidirectional, bool):
            bidirectional = torch.full(
                (edge_count,),
                raw_bidirectional,
                dtype=torch.bool,
                device=target,
            )
        else:
            bidirectional = _structural_tensor(
                raw_bidirectional,
                name=f"components[{graph_id}].edge_bidirectional",
                device=target,
                dtype=torch.bool,
            )
            if bidirectional.numel() != edge_count:
                raise ValueError(
                    f"components[{graph_id}].edge_bidirectional must align with edges"
                )

        if validate and edge_count:
            if tensor_scalar_bool(torch.any(src < 0)) or tensor_scalar_bool(
                torch.any(src >= node_count)
            ):
                raise ValueError(f"components[{graph_id}].edge_src is out of range")
            if tensor_scalar_bool(torch.any(dst < 0)) or tensor_scalar_bool(
                torch.any(dst >= node_count)
            ):
                raise ValueError(f"components[{graph_id}].edge_dst is out of range")

        total_nodes = require_int32_capacity(
            "packed topology node count", total_nodes + node_count
        )
        total_edges = require_int32_capacity(
            "packed topology logical edge count", total_edges + edge_count
        )
        component_arc_count = edge_count + int(bidirectional.sum().item())
        total_arcs = require_int32_capacity(
            "packed topology arc count", total_arcs + component_arc_count
        )
        node_counts.append(node_count)
        edge_counts.append(edge_count)
        local_src_parts.append(src)
        local_dst_parts.append(dst)
        bidirectional_parts.append(bidirectional)

    graph_count = require_int32_capacity("graph count", len(normalised))
    graph_node_ptr = torch.tensor(
        [0, *torch.tensor(node_counts, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device=target,
    )
    graph_edge_ptr = torch.tensor(
        [0, *torch.tensor(edge_counts, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device=target,
    )

    if total_edges:
        node_offsets = graph_node_ptr[:-1].to(torch.int64)
        graph_ids_i64 = torch.arange(graph_count, device=target, dtype=torch.int64)
        edge_counts_tensor = torch.tensor(
            edge_counts, device=target, dtype=torch.int64
        )
        edge_graph_long = torch.repeat_interleave(
            graph_ids_i64, edge_counts_tensor
        )
        edge_src = (
            torch.cat(local_src_parts) + node_offsets[edge_graph_long]
        ).to(torch.int32)
        edge_dst = (
            torch.cat(local_dst_parts) + node_offsets[edge_graph_long]
        ).to(torch.int32)
        edge_bidirectional = torch.cat(bidirectional_parts).contiguous()
        edge_graph_ids = edge_graph_long.to(torch.int32)
    else:
        edge_src = torch.empty(0, dtype=torch.int32, device=target)
        edge_dst = torch.empty(0, dtype=torch.int32, device=target)
        edge_bidirectional = torch.empty(0, dtype=torch.bool, device=target)
        edge_graph_ids = torch.empty(0, dtype=torch.int32, device=target)

    graph_ids_i32 = torch.arange(graph_count, device=target, dtype=torch.int32)
    node_graph_ids = torch.repeat_interleave(
        graph_ids_i32,
        torch.tensor(node_counts, device=target, dtype=torch.int64),
    )

    if total_arcs:
        arcs_per_edge = 1 + edge_bidirectional.to(torch.int64)
        logical_edge_ids = torch.repeat_interleave(
            torch.arange(total_edges, device=target, dtype=torch.int64),
            arcs_per_edge,
        )
        edge_arc_ptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=target),
                torch.cumsum(arcs_per_edge, dim=0),
            )
        )
        direction = (
            torch.arange(total_arcs, dtype=torch.int64, device=target)
            - edge_arc_ptr[logical_edge_ids]
        )
        forward = direction == 0
        logical_src = edge_src[logical_edge_ids]
        logical_dst = edge_dst[logical_edge_ids]
        unsorted_src = torch.where(forward, logical_src, logical_dst)
        unsorted_dst = torch.where(forward, logical_dst, logical_src)
        csr_order = torch.argsort(
            unsorted_src.to(torch.int64), stable=True
        )
        adjacency_dst = unsorted_dst[csr_order].to(torch.int32).contiguous()
        adjacency_edge_id = logical_edge_ids[csr_order].to(torch.int32).contiguous()
        sorted_src = unsorted_src[csr_order].to(torch.int64)
        row_counts = torch.bincount(sorted_src, minlength=total_nodes)
        row_ptr64 = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=target),
                torch.cumsum(row_counts, dim=0),
            )
        )
        if int(row_ptr64[-1].item()) > INT32_MAX:
            raise OverflowError("CSR cumulative offsets exceed int32 capacity")
        adjacency_row_ptr = row_ptr64.to(torch.int32)
    else:
        adjacency_dst = torch.empty(0, dtype=torch.int32, device=target)
        adjacency_edge_id = torch.empty(0, dtype=torch.int32, device=target)
        adjacency_row_ptr = torch.zeros(
            total_nodes + 1, dtype=torch.int32, device=target
        )

    graph_arc_ptr = adjacency_row_ptr[graph_node_ptr.to(torch.int64)].contiguous()
    topology = PackedTopology(
        edge_src=edge_src.contiguous(),
        edge_dst=edge_dst.contiguous(),
        edge_bidirectional=edge_bidirectional.contiguous(),
        adjacency_row_ptr=adjacency_row_ptr.contiguous(),
        adjacency_dst=adjacency_dst.contiguous(),
        adjacency_edge_id=adjacency_edge_id.contiguous(),
        graph_node_ptr=graph_node_ptr.contiguous(),
        graph_edge_ptr=graph_edge_ptr.contiguous(),
        graph_arc_ptr=graph_arc_ptr,
        node_graph_ids=node_graph_ids.contiguous(),
        edge_graph_ids=edge_graph_ids.contiguous(),
        execution_policy=policy,
        schema=resolved_schema,
    )
    if validate:
        topology.validate()
    return topology
