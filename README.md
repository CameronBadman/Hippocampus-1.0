# Hippocampus Graph

Hippocampus Graph is a PyTorch-native data layer for graph topology and
variable-length manifolds. It keeps reusable CSR topology separate from each
differentiable value snapshot, supports CPU and CUDA storage without implicit
precision changes, and prepares attention-ready layouts without invoking an
attention implementation.

The milestone includes:

- reusable, int32 `PackedTopology` snapshots with canonical outbound CSR;
- differentiable ragged and dense-candidate manifold packing;
- duplicate-preserving gather with complete shuffle and source provenance;
- CUDA varlen, power-of-two, and single padded working layouts;
- occurrence-preserving frontier expansion; and
- stable segmented reductions and row-preserving transforms.

Attention execution, cross-attention pairing, serialization, neural-output
caching across optimizer steps, custom kernels, CUDA Graph capture, and
multi-GPU sharding are intentionally outside this package.

## Install

Python 3.10 or newer and PyTorch 2.2 or newer are required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
pytest
```

For CUDA, install the PyTorch build appropriate for the host driver before
installing this package.

## Static graph construction

`GraphBuilder.compile()` is the fixture and inference convenience path. Its
default is deliberately CPU FP32, even if the supplied tensors have another
dtype.

```python
import torch

from hippocampus import GraphBuilder, GraphSchema

schema = GraphSchema(summary_dim=4, context_dim=8, edge_dim=6)
builder = GraphBuilder(schema)

left = builder.add_node(torch.randn(4), torch.randn(2, 8))
right = builder.add_node(torch.randn(3, 4))  # empty context is valid
builder.add_edge(left, right, torch.randn(2, 6), bidirectional=True)

graph = builder.compile()
```

A bidirectional edge is one logical edge with one shared edge manifold and two
traversal arcs. Use two directed logical edges when forward and reverse content
must differ.

## Training with reusable topology

Create and move topology once. Repack encoder outputs after every applicable
forward pass; a packed value snapshot preserves that forward's autograd graph
and must not be reused as a neural-output cache across optimizer steps.

```python
import torch

from hippocampus import (
    GraphSchema,
    RaggedManifoldComponents,
    TopologyComponent,
    pack_graph_from_topology,
    pack_topology,
)

schema = GraphSchema(summary_dim=4, context_dim=8, edge_dim=6)
topology = pack_topology(
    [
        TopologyComponent(
            node_count=2,
            edge_src=[0],
            edge_dst=[1],
            edge_bidirectional=True,
            schema=schema,
        )
    ]
).to("cuda")

# Recreate these tensors and this PackedGraph on each writer forward.
summaries = torch.randn(3, 4, device="cuda", requires_grad=True)
contexts = torch.randn(2, 8, device="cuda", requires_grad=True)
edges = torch.randn(2, 6, device="cuda", requires_grad=True)

graph = pack_graph_from_topology(
    topology,
    RaggedManifoldComponents(summaries, [0, 1, 3]),
    RaggedManifoldComponents(contexts, [0, 0, 2]),
    RaggedManifoldComponents(edges, [0, 2]),
)
```

With no `PackConfig`, tensor-level packing requires and preserves one common
source device and value dtype. Mixed source devices or dtypes raise. An
explicit configuration may move or cast values, and those operations remain
differentiable:

```python
from hippocampus import PackConfig

config = PackConfig.cuda_bf16(pin_cpu_staging=True)
```

Explicit CUDA BF16 validates hardware support and never falls back to FP16.
Pinned staging is invalid for a CPU target. A non-blocking CPU-to-CUDA transfer
only has overlap guarantees when its CPU source buffer is pinned.

## Gather and layouts

Gather retains owner occurrence order and duplicates. A row shuffle is local
to each selected manifold and moves values, presence gates, and provenance
together.

```python
selected = graph.summaries.gather(
    torch.tensor([1, 0, 1], device=graph.device),
    shuffle=True,
    generator=torch.Generator(device=graph.device).manual_seed(123),
)

varlen = selected.layout("auto", supports_cuda_varlen=True)
padded = selected.layout("power_of_two")
single = selected.layout("single")
```

`auto` selects varlen only for CUDA values when the consumer explicitly
declares support. Layout objects are data representations, not attention calls.
Empty owners remain represented through separate owner and selection-position
mappings.

## Frontier expansion

```python
expansion = graph.expand_frontier(
    torch.tensor([0, 0, 1], device=graph.device)
)
```

Expansion preserves frontier duplicates and CSR order. It returns arc IDs,
logical-edge IDs, source and destination nodes, frontier occurrence positions,
and cumulative arc offsets.

## Segmented operations

Operations accept either offsets or row-owner IDs and return values plus a
segment-aligned validity mask.

```python
from hippocampus import segment_logsumexp, segment_softmax

reduced = segment_logsumexp(selected.values, selected.offsets)
weights = segment_softmax(selected.values, selected.offsets)
```

Sum, mean, max, log-sum-exp, weighted sum, and weighted mean return
`SegmentReduction`. Softmax and broadcast return `SegmentTransform`. Empty
segments and zero-total-weight segments are invalid and receive a configurable
fill (zero by default). FP16 and BF16 reductions accumulate in FP32 by default
and return the requested output dtype.

## Determinism

The library does not change global PyTorch settings:

```python
torch.use_deterministic_algorithms(True)
```

After enabling that state, construct `ExecutionPolicy("deterministic")`.
Deterministic row permutation additionally requires an explicitly seeded
generator. Replay guarantees assume identical hardware, software versions,
dtype, seeds, and execution configuration.

## Snapshot IDs and aliasing

Node, logical-edge, and arc IDs are global only within one packed topology
snapshot. Arc ID is its final CSR position. Repacking, graph-batch reordering,
compaction, or topology reconstruction may assign different IDs.

Normal packing owns its structural and value storage while preserving autograd.
`PackedTopology.unsafe_from_packed()` and
`PackedGraph.unsafe_from_packed()` are deeply validated zero-copy entry points;
their tensors remain externally mutable and aliased by design.

See [architecture.md](docs/architecture.md) for the detailed contracts and
[cached-lengths.md](docs/cached-lengths.md) for the CUDA benchmark decision.

