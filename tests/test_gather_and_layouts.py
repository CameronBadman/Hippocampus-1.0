from __future__ import annotations

import pytest
import torch

from hippocampus import (
    DenseCandidateComponents,
    ExecutionPolicy,
    GraphSchema,
    ManifoldLayoutConfig,
    PowerOfTwoManifoldBatch,
    RaggedManifoldComponents,
    SingleManifoldBatch,
    TopologyComponent,
    VarlenManifoldBatch,
    pack_graph_from_topology,
    pack_topology,
)


def _context_family(*, presence: bool = True):
    schema = GraphSchema(2, 2, 2, allow_empty_edge_manifolds=True)
    topology = pack_topology(
        [TopologyComponent(5, [], [], False, schema)]
    )
    values = torch.arange(20.0).reshape(10, 2).requires_grad_()
    gates = torch.arange(10.0).requires_grad_() if presence else None
    graph = pack_graph_from_topology(
        topology,
        torch.ones(5, 2),
        RaggedManifoldComponents(
            values,
            [0, 1, 1, 4, 6, 10],
            gates,
        ),
        RaggedManifoldComponents(torch.empty(0, 2), [0]),
    )
    return graph.contexts, values, gates


def test_gather_preserves_duplicates_and_mapping_definitions() -> None:
    family, values, gates = _context_family()
    gathered = family.gather(torch.tensor([2, 1, 2, 0]))

    assert gathered.offsets.tolist() == [0, 3, 3, 6, 7]
    assert gathered.lengths.tolist() == [3, 0, 3, 1]
    assert gathered.owner_ids.tolist() == [2, 1, 2, 0]
    assert gathered.selection_positions.tolist() == [0, 1, 2, 3]
    assert gathered.base_source_row_ids.tolist() == [1, 2, 3, 1, 2, 3, 0]
    assert gathered.forward_shuffle.tolist() == list(range(7))
    assert gathered.inverse_shuffle.tolist() == list(range(7))
    assert torch.equal(
        gathered.source_row_ids,
        gathered.base_source_row_ids[gathered.forward_shuffle],
    )

    (gathered.values.sum() + gathered.presence.sum()).backward()
    assert values.grad[:, 0].tolist() == [1, 2, 2, 2, 0, 0, 0, 0, 0, 0]
    assert gates.grad.tolist() == [1, 2, 2, 2, 0, 0, 0, 0, 0, 0]


def test_shuffle_is_segment_local_and_moves_presence_and_provenance() -> None:
    schema = GraphSchema(2, 2, 2)
    topology = pack_topology(
        [TopologyComponent(2, [0], [1], False, schema)]
    )
    candidates = torch.arange(16.0).reshape(1, 8, 2)
    valid = torch.tensor([[True, False, True, True, False, True, True, False]])
    presence = torch.arange(8.0).reshape(1, 8)
    graph = pack_graph_from_topology(
        topology,
        torch.ones(2, 2),
        RaggedManifoldComponents(torch.empty(0, 2), [0, 0, 0]),
        DenseCandidateComponents(candidates, valid, presence),
    )
    gathered = graph.edges.gather(
        torch.tensor([0, 0]),
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    assert sorted(gathered.forward_shuffle[:5].tolist()) == list(range(5))
    assert sorted(gathered.forward_shuffle[5:].tolist()) == list(range(5, 10))
    assert torch.equal(
        gathered.values,
        graph.edges.values[gathered.source_row_ids],
    )
    assert torch.equal(
        gathered.presence,
        graph.edges.presence[gathered.source_row_ids],
    )
    assert torch.equal(
        gathered.candidate_slot_ids,
        graph.edges.candidate_slot_ids[gathered.source_row_ids],
    )
    assert torch.equal(
        gathered.base_source_row_ids,
        gathered.source_row_ids[gathered.inverse_shuffle],
    )


def test_deterministic_shuffle_requires_generator_and_replays() -> None:
    torch.use_deterministic_algorithms(True)
    try:
        policy = ExecutionPolicy("deterministic")
        schema = GraphSchema(2, 2, 2, allow_empty_edge_manifolds=True)
        topology = pack_topology(
            [TopologyComponent(2, [], [], False, schema)],
            execution_policy=policy,
        )
        graph = pack_graph_from_topology(
            topology,
            torch.ones(2, 2),
            RaggedManifoldComponents(
                torch.arange(12.0).reshape(6, 2), [0, 3, 6]
            ),
            RaggedManifoldComponents(torch.empty(0, 2), [0]),
        )
        with pytest.raises(RuntimeError, match="seeded generator"):
            graph.contexts.gather(torch.tensor([0, 1]), shuffle=True)
        first = graph.contexts.gather(
            torch.tensor([0, 1]),
            shuffle=True,
            generator=torch.Generator().manual_seed(91),
        )
        second = graph.contexts.gather(
            torch.tensor([0, 1]),
            shuffle=True,
            generator=torch.Generator().manual_seed(91),
        )
        assert torch.equal(first.forward_shuffle, second.forward_shuffle)
        assert torch.equal(first.values, second.values)
    finally:
        torch.use_deterministic_algorithms(False)


def test_empty_and_all_empty_gather_contracts() -> None:
    family, _, _ = _context_family()
    empty = family.gather(torch.empty(0, dtype=torch.int64))
    assert empty.values.shape == (0, 2)
    assert empty.offsets.tolist() == [0]
    assert empty.owner_ids.shape == (0,)
    assert empty.forward_shuffle.shape == (0,)
    assert empty.presence.shape == (0,)

    all_empty = family.gather(torch.tensor([1, 1, 1]))
    assert all_empty.values.shape == (0, 2)
    assert all_empty.offsets.tolist() == [0, 0, 0, 0]
    assert all_empty.lengths.tolist() == [0, 0, 0]
    assert all_empty.owner_ids.tolist() == [1, 1, 1]
    assert all_empty.selection_positions.tolist() == [0, 1, 2]
    assert all_empty.presence.shape == (0,)


def test_varlen_layout_retains_nonempty_and_empty_mappings() -> None:
    family, _, _ = _context_family()
    gathered = family.gather(torch.tensor([2, 1, 4, 1, 0]))
    layout = gathered.layout("varlen")
    assert isinstance(layout, VarlenManifoldBatch)
    assert layout.lengths.tolist() == [3, 4, 1]
    assert layout.cu_seqlens.tolist() == [0, 3, 7, 8]
    assert layout.max_seqlen == 4
    assert layout.owner_ids.tolist() == [2, 4, 0]
    assert layout.selection_positions.tolist() == [0, 2, 4]
    assert layout.empty_owner_ids.tolist() == [1, 1]
    assert layout.empty_selection_positions.tolist() == [1, 3]
    assert layout.values.data_ptr() == gathered.values.data_ptr()
    layout.validate()


def test_empty_varlen_layout_contract() -> None:
    family, _, _ = _context_family()
    layout = family.gather(torch.tensor([1, 1])).layout("varlen")
    assert layout.values.shape == (0, 2)
    assert layout.cu_seqlens.tolist() == [0]
    assert layout.lengths.shape == (0,)
    assert layout.max_seqlen == 0
    assert layout.owner_ids.shape == (0,)
    assert layout.empty_owner_ids.tolist() == [1, 1]
    assert layout.presence.shape == (0,)


def test_power_of_two_buckets_split_only_between_manifolds() -> None:
    family, _, _ = _context_family()
    gathered = family.gather(torch.tensor([0, 2, 2, 4, 3, 1]))
    layout = gathered.layout(
        "power_of_two",
        config=ManifoldLayoutConfig(target_padded_rows_per_launch=8),
    )
    assert isinstance(layout, PowerOfTwoManifoldBatch)
    assert [
        (bucket.batch_size, bucket.padded_rows_per_manifold)
        for bucket in layout.buckets
    ] == [(1, 1), (1, 2), (2, 4), (1, 4)]
    assert [bucket.owner_ids.tolist() for bucket in layout.buckets] == [
        [0],
        [3],
        [2, 2],
        [4],
    ]
    assert layout.empty_owner_ids.tolist() == [1]
    assert sum(bucket.mask.sum().item() for bucket in layout.buckets) == 13
    layout.validate()


def test_power_of_two_uses_exact_singleton_when_rounding_exceeds_target() -> None:
    family, _, _ = _context_family()
    layout = family.gather(torch.tensor([4])).layout(
        "power_of_two",
        config=ManifoldLayoutConfig(target_padded_rows_per_launch=6),
    )
    assert len(layout.buckets) == 1
    bucket = layout.buckets[0]
    assert bucket.batch_size == 1
    assert bucket.lengths.tolist() == [4]
    assert bucket.padded_rows_per_manifold == 4

    layout = family.gather(torch.tensor([2])).layout(
        "power_of_two",
        config=ManifoldLayoutConfig(target_padded_rows_per_launch=2),
    )
    assert layout.buckets[0].lengths.tolist() == [3]
    assert layout.buckets[0].padded_rows_per_manifold == 3


def test_valid_singleton_may_exceed_soft_target() -> None:
    family, _, _ = _context_family()
    layout = family.gather(torch.tensor([4])).layout(
        "power_of_two",
        config=ManifoldLayoutConfig(target_padded_rows_per_launch=2),
    )
    assert layout.buckets[0].padded_rows_per_manifold == 4


def test_single_layout_and_target_limit() -> None:
    family, _, _ = _context_family()
    gathered = family.gather(torch.tensor([2, 1, 4]))
    layout = gathered.layout("single")
    assert isinstance(layout, SingleManifoldBatch)
    assert layout.values.shape == (2, 4, 2)
    assert layout.mask.tolist() == [
        [True, True, True, False],
        [True, True, True, True],
    ]
    assert layout.empty_owner_ids.tolist() == [1]
    layout.validate()
    with pytest.raises(ValueError, match="exceeding"):
        gathered.layout(
            "single",
            config=ManifoldLayoutConfig(
                target_padded_rows_per_launch=7
            ),
        )


def test_all_empty_padded_layout_contracts() -> None:
    family, _, _ = _context_family()
    gathered = family.gather(torch.tensor([1, 1]))
    power = gathered.layout("power_of_two")
    single = gathered.layout("single")
    assert power.buckets == ()
    assert power.empty_owner_ids.tolist() == [1, 1]
    assert single.values.shape == (0, 0, 2)
    assert single.mask.shape == (0, 0)
    assert single.lengths.shape == (0,)
    assert single.empty_selection_positions.tolist() == [0, 1]
    assert single.presence.shape == (0, 0)


def test_auto_policy_is_power_of_two_on_cpu() -> None:
    family, _, _ = _context_family()
    layout = family.gather(torch.tensor([0, 2])).layout(
        "auto", supports_cuda_varlen=True
    )
    assert isinstance(layout, PowerOfTwoManifoldBatch)


def test_layout_hard_manifold_limit_and_single_target() -> None:
    family, _, _ = _context_family()
    gathered = family.gather(torch.tensor([4]))
    with pytest.raises(ValueError, match="max_rows_per_manifold"):
        gathered.layout(
            "varlen",
            config=ManifoldLayoutConfig(max_rows_per_manifold=3),
        )


def test_invalid_owner_ids_and_trusted_hot_path() -> None:
    family, _, _ = _context_family()
    with pytest.raises(IndexError, match="outside"):
        family.gather(torch.tensor([5]))
    trusted = family.gather(torch.tensor([0]), validate_ids=False)
    assert trusted.lengths.tolist() == [1]
