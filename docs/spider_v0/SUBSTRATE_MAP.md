# Spider v0 substrate map

This map records the packed-graph API verified before Spider v0 development.
It describes the implementation at commit `e71212a`.

## Verification baseline

- Complete suite: `62 passed`
- CUDA substrate suite: `4 passed`
- Device: NVIDIA GeForce RTX 5070 Ti
- PyTorch: `2.13.0+cu130`
- CUDA runtime reported by PyTorch: `13.0`
- CUDA BF16 support: available

Commands:

```bash
.venv/bin/pytest -q
.venv/bin/pytest -q tests/test_cuda.py
```

## Public structural API

### `GraphSchema`

Defined in `hippocampus.config`.

```python
GraphSchema(
    summary_dim: int,
    context_dim: int,
    edge_dim: int,
    allow_empty_edge_manifolds: bool = False,
)
```

It is the sole source for persistent family widths and empty-manifold
invariants:

- every node summary is non-empty;
- node contexts may be empty;
- edge manifolds may be empty only with explicit schema opt-in.

### `PackedTopology`

Defined in `hippocampus.topology`. Persistent tensors are int32 except
`edge_bidirectional`, which is bool.

Logical topology:

- `edge_src`
- `edge_dst`
- `edge_bidirectional`

Canonical outbound CSR:

- `adjacency_row_ptr`
- `adjacency_dst`
- `adjacency_edge_id`

Batch mappings:

- `graph_node_ptr`
- `graph_edge_ptr`
- `graph_arc_ptr`
- `node_graph_ids`
- `edge_graph_ids`

Useful methods and properties:

- `to(device, non_blocking=False)`
- `pin_memory()`
- `validate()`
- `expand_frontier(frontier_node_ids, validate_ids=True)`
- `node_count`, `edge_count`, `arc_count`, `graph_count`, `device`
- `unsafe_from_packed(...)`

Node, logical-edge, and arc IDs are snapshot-local. Arc ID is the final CSR
position and is therefore a stable final tie-breaker for one topology
snapshot.

### `FrontierExpansion`

Returned by `PackedTopology.expand_frontier` and delegated by
`PackedGraph.expand_frontier`.

Fields:

- `arc_ids`
- `edge_ids`
- `source_node_ids`
- `destination_node_ids`
- `frontier_positions`
- `arc_offsets`

Expansion preserves caller frontier occurrence order, duplicate node
occurrences, and canonical CSR order. Empty input returns `arc_offsets=[0]`;
isolated nodes produce repeated offsets.

Spider decision: this is the only topology expansion mechanism used in the
controller hot path. No second Python adjacency representation will be
maintained during traversal or training.

## Public value-snapshot API

### `PackedGraph`

Defined in `hippocampus.graph`.

Fields:

- `topology: PackedTopology`
- `summaries: PackedManifoldFamily`
- `contexts: PackedManifoldFamily`
- `edges: PackedManifoldFamily`
- `schema: GraphSchema`
- `resolved_pack_config: ResolvedPackConfig`

Aliases `summary`, `context`, and `edge` are available. A packed graph delegates
`expand_frontier` to its topology. Topology is reused; neural value snapshots
are rebuilt on each applicable forward.

### `PackedManifoldFamily`

Defined in `hippocampus.manifold`.

Persistent fields:

- `values[total_rows, width]`
- `offsets[owner_count + 1]` int32
- `row_owner_ids[total_rows]` int32
- `owner_graph_ids[owner_count]` int32
- cached `lengths[owner_count]` int32
- optional differentiable `presence[total_rows]`
- optional dense-candidate `candidate_slot_ids[total_rows]`
- `resolved_pack_config`

Important methods:

```python
gather(
    owner_ids,
    *,
    shuffle=False,
    generator=None,
    validate_ids=True,
) -> RaggedManifoldBatch
```

and:

```python
layout(owner_ids, policy="auto", ...)
```

### Differentiable component packing

`RaggedManifoldComponents(values, offsets, presence=None)` accepts compact
writer outputs directly.

`DenseCandidateComponents(values, valid_mask, presence=None)` accepts
`[owners, max_slots, width]` candidates and performs vectorized owner-major,
slot-major compaction. The bool mask is structural; presence remains
differentiable. Dense slot provenance is retained and never injected into
features.

Normal packing owns new value storage without detaching the source autograd
graph. Default tensor-level packing preserves a single common value dtype and
device and rejects mixed inputs.

### Packing entry points

```python
pack_topology(
    components,
    *,
    device=None,
    execution_policy=None,
    validate=True,
    schema=None,
) -> PackedTopology
```

```python
pack_graph_from_topology(
    topology,
    summaries,
    contexts,
    edges,
    *,
    pack_config=None,
    execution_policy=None,
    validate=True,
    schema=None,
) -> PackedGraph
```

`pack_graph_from_topology` never moves topology implicitly. Value tensors must
resolve to the topology device.

`pack_graph_batch_from_components` and `GraphBuilder.compile` are static
convenience paths. Spider dataset collation will use the two-stage API so that
one topology snapshot can be reused across value forwards.

## Gather and provenance

`RaggedManifoldBatch` fields include:

- compact `values` and optional `presence`;
- `offsets`, `lengths`, selected `owner_ids`;
- `selection_positions`, `owner_graph_ids`;
- per-row `row_owner_ids`, `row_selection_positions`;
- `base_source_row_ids`;
- `forward_shuffle`, `inverse_shuffle`, `source_row_ids`;
- optional `candidate_slot_ids`.

The mapping definitions are:

```text
shuffled = base[forward_shuffle]
base = shuffled[inverse_shuffle]
source_row_ids = base_source_row_ids[forward_shuffle]
```

Selections preserve occurrence order and duplicates. Duplicate advanced
indexing accumulates gradients into source rows.

Spider decision: hypothesis occurrences are allowed to select the same
persistent owner repeatedly. Candidate-to-hypothesis alignment uses
`frontier_positions`; manifold alignment uses gather occurrence positions.

## Working manifold layouts

`RaggedManifoldBatch.layout` and `build_manifold_layout` support:

- `VarlenManifoldBatch`
- `PowerOfTwoManifoldBatch` containing `PaddedManifoldBucket` objects
- `SingleManifoldBatch`

`auto` selects varlen only for CUDA values when the consumer explicitly
declares `supports_cuda_varlen=True`; otherwise it selects power-of-two
padding.

Varlen fields include compact values, int32 `cu_seqlens`, positive lengths,
`max_seqlen`, non-empty owner mappings, separate empty-owner mappings, and all
row provenance.

Padded buckets include:

- `values[batch, padded_rows, width]`
- bool `mask[batch, padded_rows]`
- lengths and selected owner mappings
- padded source-row and row-owner mappings
- optional padded presence and dense-slot provenance.

Layouts never invoke attention. Spider will initially consume padded layouts
through PyTorch SDPA. Varlen is enabled only behind an explicit backend
capability/equivalence check.

## Segmented tensor primitives

Defined in `hippocampus.segmented`. Operations accept offsets or row-owner IDs.

Reductions returning `SegmentReduction(values, valid_mask)`:

- `segment_sum`
- `segment_mean`
- `segment_max`
- `segment_logsumexp`
- `segment_weighted_sum`
- `segment_weighted_mean`

Transforms returning `SegmentTransform(values, valid_mask)`:

- `segment_softmax`
- `segment_broadcast`

Empty segments and zero-total-weight weighted segments are invalid and use a
configurable fill. FP16/BF16 accumulate in FP32 by default. Log-sum-exp and
softmax use stable maximum subtraction.

Spider decision: evidence pooling, per-destination normalization, and
occurrence aggregation will reuse these primitives where the operation is
actually segmented. Candidate pruning remains a stable deterministic
controller operation.

## Determinism

`ExecutionPolicy("fast" | "deterministic")` is defined in
`hippocampus.config`.

Deterministic construction requires:

```python
torch.use_deterministic_algorithms(True)
```

The package validates but never mutates that global setting. Gather shuffling
under deterministic mode additionally requires a caller-supplied seeded
generator. Stable topology ordering and deterministic segmented reference
implementations already exist.

Spider decision: controller ranking adds explicit stable keys ending in
snapshot-local arc ID and candidate occurrence index. Evaluation will run with
deterministic algorithms enabled and compare exact trace ledgers.

## Presence and provenance

Presence gates are aligned differentiable values and move with all gather and
layout permutations. Spider will use them only as multiplicative observation
strengths; they do not carry semantic labels.

Available exact provenance includes:

- owner-to-graph mappings;
- row-to-owner mappings;
- dense-candidate slot IDs;
- gather selection occurrence mappings;
- source flat-row IDs;
- arc-to-logical-edge IDs;
- frontier occurrence mappings.

No provenance ID will be projected into neural features.

## Required Spider substrate extensions

No packed-storage extension is required for Spider v0.

The substrate intentionally does not execute attention, maintain transient
hypotheses, rank sparse candidates, or store query-local evidence. Those are
Spider-layer responsibilities and will be implemented without modifying
persistent graph storage.

The only generic behavior Spider depends on beyond storage is already present:

1. duplicate-preserving frontier expansion;
2. duplicate-preserving manifold gather;
3. padded/varlen metadata construction;
4. differentiable presence/provenance movement; and
5. segmented reductions and transforms.

