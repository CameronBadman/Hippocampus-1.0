from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from hippocampus import (
    DenseCandidateComponents,
    GraphBuilder,
    GraphComponents,
    GraphSchema,
    PackConfig,
    PackedGraph,
    RaggedManifoldComponents,
    TopologyComponent,
    pack_graph_batch_from_components,
    pack_graph_from_topology,
    pack_topology,
)
from hippocampus.config import resolve_pack_config


def _basic_topology(schema: GraphSchema | None = None):
    return pack_topology(
        [TopologyComponent(2, [0], [1], False, schema)]
    )


def _families(dtype: torch.dtype = torch.float32):
    return (
        RaggedManifoldComponents(
            torch.randn(3, 3, dtype=dtype), [0, 1, 3]
        ),
        RaggedManifoldComponents(
            torch.randn(1, 2, dtype=dtype), [0, 0, 1]
        ),
        RaggedManifoldComponents(
            torch.randn(2, 4, dtype=dtype), [0, 2]
        ),
    )


def test_default_tensor_packing_preserves_common_device_and_dtype() -> None:
    schema = GraphSchema(3, 2, 4)
    topology = _basic_topology(schema)
    graph = pack_graph_from_topology(topology, *_families(torch.float64))
    assert graph.device == torch.device("cpu")
    assert graph.dtype == torch.float64
    assert graph.summaries.dtype == torch.float64
    assert graph.resolved_pack_config.execution_mode == "fast"


def test_graph_schema_must_be_explicit_or_attached_to_topology() -> None:
    topology = _basic_topology()
    with pytest.raises(ValueError, match="requires a GraphSchema"):
        pack_graph_from_topology(topology, *_families())
    graph = pack_graph_from_topology(
        topology,
        *_families(),
        schema=GraphSchema(3, 2, 4),
    )
    assert graph.schema == GraphSchema(3, 2, 4)


def test_default_packing_rejects_mixed_value_dtypes() -> None:
    schema = GraphSchema(3, 2, 4)
    topology = _basic_topology(schema)
    summaries, contexts, edges = _families()
    contexts = RaggedManifoldComponents(
        contexts.values.double(), contexts.offsets
    )
    with pytest.raises(ValueError, match="common source dtype"):
        pack_graph_from_topology(topology, summaries, contexts, edges)


def test_explicit_cast_remains_differentiable() -> None:
    schema = GraphSchema(3, 2, 4)
    topology = _basic_topology(schema)
    summaries, contexts, edges = _families(torch.float64)
    summaries.values.requires_grad_()
    graph = pack_graph_from_topology(
        topology,
        summaries,
        contexts,
        edges,
        pack_config=PackConfig(device="cpu", value_dtype=torch.float32),
    )
    graph.summaries.values.square().sum().backward()
    assert summaries.values.grad is not None
    assert summaries.values.grad.dtype == torch.float64
    assert graph.summaries.dtype == torch.float32


def test_topology_is_never_moved_implicitly() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to make a mixed-device value family")
    schema = GraphSchema(3, 2, 4)
    topology = _basic_topology(schema)
    families = tuple(
        RaggedManifoldComponents(
            family.values.cuda(),
            torch.as_tensor(family.offsets, device="cuda"),
        )
        for family in _families()
    )
    with pytest.raises(ValueError, match="call topology.to"):
        pack_graph_from_topology(topology, *families)


def test_graph_builder_defaults_to_cpu_fp32() -> None:
    schema = GraphSchema(2, 3, 4)
    builder = GraphBuilder(schema)
    left = builder.add_node(torch.ones(2, dtype=torch.float64))
    right = builder.add_node(
        torch.ones(2, dtype=torch.float16),
        torch.ones(2, 3, dtype=torch.float16),
    )
    builder.add_edge(
        left,
        right,
        torch.ones(4, dtype=torch.float64),
        bidirectional=True,
    )
    graph = builder.compile()
    assert graph.device.type == "cpu"
    assert graph.dtype == torch.float32
    assert graph.contexts.lengths.tolist() == [0, 2]
    assert graph.topology.adjacency_edge_id.tolist() == [0, 0]


def test_schema_empty_contracts() -> None:
    schema = GraphSchema(2, 2, 2)
    topology = _basic_topology(schema)
    empty_summary = RaggedManifoldComponents(
        torch.ones(1, 2), [0, 0, 1]
    )
    contexts = RaggedManifoldComponents(
        torch.empty(0, 2), [0, 0, 0]
    )
    edges = RaggedManifoldComponents(torch.empty(0, 2), [0, 0])
    with pytest.raises(ValueError, match="summaries owner 0"):
        pack_graph_from_topology(
            topology, empty_summary, contexts, torch.ones(1, 2)
        )
    with pytest.raises(ValueError, match="edges owner 0"):
        pack_graph_from_topology(
            topology, torch.ones(2, 2), contexts, edges
        )

    opted_in = GraphSchema(2, 2, 2, allow_empty_edge_manifolds=True)
    graph = pack_graph_from_topology(
        _basic_topology(opted_in),
        torch.ones(2, 2),
        contexts,
        edges,
    )
    assert graph.contexts.total_rows == 0
    assert graph.edges.total_rows == 0


def test_dense_candidate_compaction_and_provenance_are_differentiable() -> None:
    schema = GraphSchema(2, 2, 2)
    topology = _basic_topology(schema)
    dense = torch.arange(12.0).reshape(1, 6, 2).requires_grad_()
    presence = torch.linspace(0, 1, 6).reshape(1, 6).requires_grad_()
    valid = torch.tensor([[False, True, True, False, False, True]])
    graph = pack_graph_from_topology(
        topology,
        torch.ones(2, 2),
        RaggedManifoldComponents(
            torch.empty(0, 2), [0, 0, 0]
        ),
        DenseCandidateComponents(dense, valid, presence),
    )

    assert graph.edges.values.tolist() == dense.detach()[valid].tolist()
    assert graph.edges.row_owner_ids.tolist() == [0, 0, 0]
    assert graph.edges.candidate_slot_ids.tolist() == [1, 2, 5]
    assert graph.edges.candidate_dense_indices.tolist() == [
        [0, 1],
        [0, 2],
        [0, 5],
    ]
    loss = graph.edges.values.sum() + graph.edges.presence.sum()
    loss.backward()
    assert dense.grad[0, :, 0].tolist() == [0, 1, 1, 0, 0, 1]
    assert presence.grad[0].tolist() == [0, 1, 1, 0, 0, 1]
    assert not valid.requires_grad


def test_topology_reuse_only_connects_current_writer_snapshot() -> None:
    torch.manual_seed(7)
    schema = GraphSchema(2, 2, 2, allow_empty_edge_manifolds=True)
    topology = pack_topology(
        [TopologyComponent(2, [], [], False, schema)]
    )
    writer = torch.nn.Linear(3, 4, bias=False)
    first_output = writer(torch.randn(2, 3))
    first_output.retain_grad()
    first_graph = pack_graph_from_topology(
        topology,
        first_output[:, :2],
        first_output[:, 2:],
        RaggedManifoldComponents(torch.empty(0, 2), [0]),
    )
    second_output = writer(torch.randn(2, 3))
    second_output.retain_grad()
    second_graph = pack_graph_from_topology(
        topology,
        second_output[:, :2],
        second_output[:, 2:],
        RaggedManifoldComponents(torch.empty(0, 2), [0]),
    )

    assert first_graph.topology is topology
    assert second_graph.topology is topology
    (
        second_graph.summaries.values.sum()
        + second_graph.contexts.values.sum()
    ).backward()
    assert second_output.grad is not None
    assert first_output.grad is None


def test_batch_packing_retains_graph_mappings() -> None:
    schema = GraphSchema(2, 1, 3)
    components = [
        GraphComponents(
            schema,
            summaries=(torch.ones(2), torch.ones(2)),
            contexts=(torch.empty(0, 1), torch.ones(1)),
            edges=(torch.ones(3),),
            edge_src=(0,),
            edge_dst=(1,),
        ),
        GraphComponents(
            schema,
            summaries=(torch.ones(2),),
            contexts=(torch.empty(0, 1),),
            edges=(),
        ),
    ]
    graph = pack_graph_batch_from_components(components)
    assert graph.topology.graph_node_ptr.tolist() == [0, 2, 3]
    assert graph.topology.graph_edge_ptr.tolist() == [0, 1, 1]
    assert graph.summaries.owner_graph_ids.tolist() == [0, 0, 1]
    assert graph.edges.owner_graph_ids.tolist() == [0]


def test_empty_batch_requires_schema_and_has_canonical_storage() -> None:
    schema = GraphSchema(2, 1, 3)
    with pytest.raises(ValueError, match="explicit GraphSchema"):
        pack_graph_batch_from_components([])
    graph = pack_graph_batch_from_components([], schema)
    assert graph.topology.graph_count == 0
    assert graph.summaries.values.shape == (0, 2)
    assert graph.contexts.offsets.tolist() == [0]
    assert graph.edges.row_owner_ids.shape == (0,)


def test_normal_value_pack_owns_storage_but_preserves_gradients() -> None:
    schema = GraphSchema(2, 2, 2)
    topology = _basic_topology(schema)
    summaries = torch.ones(2, 2, requires_grad=True)
    graph = pack_graph_from_topology(
        topology,
        summaries,
        RaggedManifoldComponents(
            torch.empty(0, 2), [0, 0, 0]
        ),
        torch.ones(1, 2),
    )
    assert graph.summaries.values.data_ptr() != summaries.data_ptr()
    with torch.no_grad():
        summaries.add_(10)
    assert torch.all(graph.summaries.values == 1)
    graph.summaries.values.sum().backward()
    assert summaries.grad is not None


def test_unsafe_graph_constructor_aliases_families() -> None:
    schema = GraphSchema(3, 2, 4)
    graph = pack_graph_from_topology(
        _basic_topology(schema), *_families()
    )
    unsafe = PackedGraph.unsafe_from_packed(
        topology=graph.topology,
        summaries=graph.summaries,
        contexts=graph.contexts,
        edges=graph.edges,
        schema=schema,
    )
    assert unsafe.summaries.values.data_ptr() == graph.summaries.values.data_ptr()
    assert unsafe.topology is graph.topology


def test_pin_cpu_staging_is_invalid_for_cpu_target() -> None:
    with pytest.raises(ValueError, match="invalid for a CPU"):
        resolve_pack_config(
            PackConfig(device="cpu", pin_cpu_staging=True),
            source_device=torch.device("cpu"),
            source_dtype=torch.float32,
            execution_policy=None,
        )


def test_explicit_bf16_rejects_unsupported_cuda_without_fallback() -> None:
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.device", return_value=nullcontext()),
        patch("torch.cuda.is_bf16_supported", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="no FP16 fallback"):
            resolve_pack_config(
                PackConfig.cuda_bf16(),
                source_device=torch.device("cpu"),
                source_dtype=torch.float32,
                execution_policy=None,
            )
