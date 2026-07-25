# Architecture and contracts

## Snapshot lifecycle

`PackedTopology` owns immutable-by-convention structural storage:

| Structure | Meaning |
| --- | --- |
| `edge_src`, `edge_dst`, `edge_bidirectional` | Logical edges |
| `adjacency_row_ptr`, `adjacency_dst`, `adjacency_edge_id` | Outbound CSR arcs |
| `graph_node_ptr`, `graph_edge_ptr`, `graph_arc_ptr` | Graph boundaries |
| `node_graph_ids`, `edge_graph_ids` | Owner-to-source-graph mappings |

`PackedGraph` combines that topology with summary, context, and edge
`PackedManifoldFamily` snapshots. Topology may be reused across forwards and
optimizer steps. Encoder-derived families must be rebuilt from each applicable
forward because their tensors retain that forward's autograd graph.

`pack_graph_from_topology` does not move topology. The resolved value target
must exactly match `topology.device`; callers move topology once with
`PackedTopology.to`.

## Schema and configuration resolution

`GraphSchema` is the only logical source for family widths and empty-manifold
rules. It must be attached to topology components, supplied to
`pack_topology`, or explicitly passed when composing the graph.

| Workflow | Default |
| --- | --- |
| `GraphBuilder.compile()` | CPU FP32 |
| Tensor-level ragged/dense packing | Common source device and dtype |
| Explicit `PackConfig` | Requested device/dtype, differentiable move/cast |

Default tensor packing rejects mixed value devices or dtypes. Presence gates
are value tensors and participate in this check; structural masks and offsets
do not.

All resolved value snapshots record:

- concrete device and value dtype;
- persistent/ephemeral index policy;
- direct or pinned CPU staging policy; and
- fast or deterministic execution mode.

Persistent pointers, IDs, CSR arrays, owner mappings, and cached lengths are
int32 and capacity-checked. Temporary gather, permutation, and occurrence
mappings may be int64.

## Differentiable manifold inputs

`RaggedManifoldComponents` provides compact values, offsets, and optional
presence gates. `DenseCandidateComponents` provides
`[owners, max_slots, width]` values, a structural bool mask, and optional
differentiable presence gates.

Dense compaction uses deterministic owner-major, slot-major boolean indexing.
The result retains compact-row owner IDs and dense slot IDs. Slot provenance is
available to losses and analysis and is never injected into value features.

Per-owner tensor sequences are a static convenience used by `GraphBuilder`.
Neural writer paths should provide one ragged or dense-candidate tensor object,
which is compacted without a Python call per owner.

## Gather mapping definitions

For selected owner occurrences:

- `base_source_row_ids` identifies persistent flat rows before shuffling;
- `forward_shuffle` satisfies
  `shuffled = base[forward_shuffle]`;
- `inverse_shuffle` satisfies
  `base = shuffled[inverse_shuffle]`; and
- `source_row_ids = base_source_row_ids[forward_shuffle]`.

Permutations are segment-local. Owner selection order and duplicates are
preserved. Duplicate source rows use ordinary PyTorch advanced indexing, so
their gradients accumulate into the source tensor.

## Layout policy

`VarlenManifoldBatch` represents one compact sequence family. Query and
key/value families for a future cross-attention consumer must be constructed
independently, with separate cumulative offsets and maxima.

Power-of-two buckets group complete manifolds and never divide a manifold.
`target_padded_rows_per_launch` is soft: buckets split between owners. When
power-of-two rounding would exceed the target, the owner receives an
exact-length singleton. A valid singleton may itself exceed the target.

Single layout pads every non-empty selected occurrence to the selected-set
maximum and rejects an explicit target overrun.

`max_rows_per_manifold` is a hard limit for every layout policy.

## Canonical empties

For family width `d`:

| Case | Values | Structural metadata |
| --- | --- | --- |
| Empty gather selection | `[0, d]` | offsets `[0]` |
| `K` selected empty owners | `[0, d]` | `K + 1` zero offsets; owners and positions retained |
| Empty varlen | `[0, d]` | `cu_seqlens=[0]`, `max_seqlen=0` |
| Empty power-of-two | no buckets | empty owners and positions retained |
| Empty single | `[0, 0, d]` | mask `[0, 0]`; empty owners retained |

Presence outputs use corresponding empty tensors whenever the source family
carries presence.

## Deterministic execution

Fast mode uses standard vectorized PyTorch operations and may use
nondeterministic CUDA reductions.

Deterministic mode:

- checks `torch.are_deterministic_algorithms_enabled()` and raises when false;
- uses stable CSR sort and stable grouping;
- requires a caller-supplied generator for shuffled gathers; and
- uses explicit per-segment deterministic reductions rather than silently
  selecting a nondeterministic fallback.

The package never changes global PyTorch state.

## Validation and working-set safety

Deep validation runs during normal topology/value packing, builder compilation,
unsafe zero-copy construction, and explicit `validate()` calls. CUDA deep
validation can synchronize.

Hot gathers and frontier expansion trust persistent invariants and expose
optional ID validation. Layouts validate selected maximum length, cumulative
varlen capacity, padded element/byte arithmetic, launch chunk arithmetic, and
hard limits before allocation. Frontier expansion independently checks its
expanded arc count.

An arithmetic safety check is not a memory reservation: an otherwise valid
allocation can still fail with an ordinary device OOM.

