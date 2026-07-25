from __future__ import annotations

import pytest
import torch

from hippocampus import (
    GraphSchema,
    PackedTopology,
    TopologyComponent,
    pack_topology,
)


def test_canonical_csr_and_graph_boundaries() -> None:
    schema = GraphSchema(2, 3, 4)
    topology = pack_topology(
        [
            TopologyComponent(
                4,
                edge_src=[0, 0, 2, 3],
                edge_dst=[2, 1, 0, 3],
                edge_bidirectional=[False, True, True, True],
                schema=schema,
            ),
            TopologyComponent(2, [], [], False, schema),
        ]
    )

    assert topology.edge_src.tolist() == [0, 0, 2, 3]
    assert topology.edge_dst.tolist() == [2, 1, 0, 3]
    assert topology.adjacency_row_ptr.tolist() == [0, 3, 4, 5, 7, 7, 7]
    assert topology.adjacency_dst.tolist() == [2, 1, 2, 0, 0, 3, 3]
    assert topology.adjacency_edge_id.tolist() == [0, 1, 2, 1, 2, 3, 3]
    assert topology.graph_node_ptr.tolist() == [0, 4, 6]
    assert topology.graph_edge_ptr.tolist() == [0, 4, 4]
    assert topology.graph_arc_ptr.tolist() == [0, 7, 7]
    assert topology.node_graph_ids.tolist() == [0, 0, 0, 0, 1, 1]
    assert topology.edge_graph_ids.tolist() == [0, 0, 0, 0]
    assert topology.schema is schema
    assert topology.validate() is topology


def test_bidirectional_edge_is_one_logical_edge_with_two_arcs() -> None:
    schema = GraphSchema(1, 1, 1)
    shared = pack_topology(
        [TopologyComponent(2, [0], [1], [True], schema)]
    )
    distinct = pack_topology(
        [TopologyComponent(2, [0, 1], [1, 0], [False, False], schema)]
    )

    assert (shared.edge_count, shared.arc_count) == (1, 2)
    assert shared.adjacency_edge_id.tolist() == [0, 0]
    assert (distinct.edge_count, distinct.arc_count) == (2, 2)
    assert distinct.adjacency_edge_id.tolist() == [0, 1]


def test_self_loop_bidirectional_edge_retains_both_arc_occurrences() -> None:
    topology = pack_topology([TopologyComponent(1, [0], [0], True)])
    assert topology.adjacency_row_ptr.tolist() == [0, 2]
    assert topology.adjacency_dst.tolist() == [0, 0]
    assert topology.adjacency_edge_id.tolist() == [0, 0]


def test_frontier_expansion_preserves_occurrences_and_isolated_nodes() -> None:
    topology = pack_topology(
        [
            TopologyComponent(
                4,
                [0, 0, 2],
                [2, 1, 0],
                [False, True, True],
            ),
            TopologyComponent(1),
        ]
    )
    expansion = topology.expand_frontier(
        torch.tensor([0, 4, 0], dtype=torch.int32)
    )

    assert expansion.arc_offsets.tolist() == [0, 3, 3, 6]
    assert expansion.arc_ids.tolist() == [0, 1, 2, 0, 1, 2]
    assert expansion.edge_ids.tolist() == [0, 1, 2, 0, 1, 2]
    assert expansion.source_node_ids.tolist() == [0, 0, 0, 0, 0, 0]
    assert expansion.destination_node_ids.tolist() == [2, 1, 2, 2, 1, 2]
    assert expansion.frontier_positions.tolist() == [0, 0, 0, 2, 2, 2]


def test_empty_frontier_has_canonical_shapes() -> None:
    topology = pack_topology([TopologyComponent(2)])
    expansion = topology.expand_frontier(
        torch.empty(0, dtype=torch.int64)
    )
    assert expansion.arc_offsets.tolist() == [0]
    assert expansion.arc_ids.shape == (0,)
    assert expansion.edge_ids.dtype == torch.int32
    assert expansion.frontier_positions.dtype == torch.int64


def test_frontier_id_validation_can_be_disabled_for_trusted_path() -> None:
    topology = pack_topology([TopologyComponent(2, [0], [1])])
    with pytest.raises(IndexError, match="outside"):
        topology.expand_frontier(torch.tensor([2]), validate_ids=True)
    trusted = topology.expand_frontier(
        torch.tensor([0]), validate_ids=False
    )
    assert trusted.destination_node_ids.tolist() == [1]


def test_topology_ids_are_snapshot_local_after_batch_reordering() -> None:
    first = TopologyComponent(2, [0], [1])
    second = TopologyComponent(3, [2], [0])
    original = pack_topology([first, second])
    reordered = pack_topology([second, first])

    assert original.edge_src.tolist() == [0, 4]
    assert reordered.edge_src.tolist() == [2, 3]
    assert original.node_graph_ids.tolist() == [0, 0, 1, 1, 1]
    assert reordered.node_graph_ids.tolist() == [0, 0, 0, 1, 1]
    assert original.adjacency_dst.tolist() != reordered.adjacency_dst.tolist()


def test_normal_topology_pack_owns_input_storage() -> None:
    src = torch.tensor([0], dtype=torch.int64)
    dst = torch.tensor([1], dtype=torch.int64)
    topology = pack_topology([TopologyComponent(2, src, dst)])
    src[0] = 1
    dst[0] = 0
    assert topology.edge_src.tolist() == [0]
    assert topology.edge_dst.tolist() == [1]


def test_unsafe_topology_constructor_aliases_and_deeply_validates() -> None:
    packed = pack_topology([TopologyComponent(2, [0], [1], True)])
    tensors = {
        name: getattr(packed, name).clone()
        for name in (
            "edge_src",
            "edge_dst",
            "edge_bidirectional",
            "adjacency_row_ptr",
            "adjacency_dst",
            "adjacency_edge_id",
            "graph_node_ptr",
            "graph_edge_ptr",
            "graph_arc_ptr",
            "node_graph_ids",
            "edge_graph_ids",
        )
    }
    unsafe = PackedTopology.unsafe_from_packed(**tensors)
    assert unsafe.edge_src.data_ptr() == tensors["edge_src"].data_ptr()
    tensors["edge_src"][0] = 1
    assert unsafe.edge_src.item() == 1
    with pytest.raises(ValueError, match="endpoints"):
        unsafe.validate()

    bad = dict(tensors)
    bad["edge_src"] = tensors["edge_src"].to(torch.int64)
    with pytest.raises(TypeError, match="int32"):
        PackedTopology.unsafe_from_packed(**bad)


def test_empty_topology_batch_is_well_formed() -> None:
    topology = pack_topology([])
    assert topology.graph_count == 0
    assert topology.graph_node_ptr.tolist() == [0]
    assert topology.graph_edge_ptr.tolist() == [0]
    assert topology.graph_arc_ptr.tolist() == [0]
    assert topology.adjacency_row_ptr.tolist() == [0]
    topology.validate()


def test_topology_rejects_cross_graph_or_out_of_range_local_endpoint() -> None:
    with pytest.raises(ValueError, match="out of range"):
        pack_topology([TopologyComponent(2, [0], [2])])


def test_topology_same_device_move_reuses_snapshot() -> None:
    topology = pack_topology([TopologyComponent(1)])
    assert topology.to("cpu") is topology

