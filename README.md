# Hippocampus Graph

Hippocampus Graph is a PyTorch-native packed graph layer plus Spider v0, a
small recurrent interpreter for exact synthetic graph programs. The data layer
keeps reusable CSR topology separate from each differentiable value snapshot,
supports CPU and CUDA storage without implicit precision changes, and prepares
attention-ready manifold layouts.

The milestone includes:

- reusable, int32 `PackedTopology` snapshots with canonical outbound CSR;
- differentiable ragged and dense-candidate manifold packing;
- duplicate-preserving gather with complete shuffle and source provenance;
- CUDA varlen, power-of-two, and single padded working layouts;
- occurrence-preserving frontier expansion; and
- stable segmented reductions and row-preserving transforms.

Spider v0 adds an exact four-family benchmark, frozen exchangeable renderer,
pooled/flat/recurrent models, position-free padded SDPA, a deterministic sparse
controller, staged training, and ID/OOD evaluation. Natural-language writers,
custom kernels, reinforcement learning, persistence, and multi-GPU sharding
remain out of scope.

## Install

Python 3.10 or newer and PyTorch 2.2 or newer are required.

```bash
uv sync --extra test
.venv/bin/pytest -q
```

An ordinary editable `pip install -e ".[test]"` is also supported.

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

## Spider v0

Generate the deterministic split manifests and leakage diagnostic:

```bash
.venv/bin/python scripts/generate_spider_dataset.py
```

Run the CPU/CUDA-capable tiny-overfit configuration:

```bash
.venv/bin/python scripts/train_spider_v0.py \
  --config configs/spider_v0/tiny_overfit.json \
  --device cuda --dtype float32
```

Run the pre-registered experiment matrix:

```bash
.venv/bin/python scripts/run_spider_autoresearch.py \
  --steps 60 --train-cases 48 --eval-cases 8 \
  --device cuda --dtype float32
```

Evaluate a non-sealed split:

```bash
.venv/bin/python scripts/evaluate_spider_v0.py \
  --config artifacts/spider_v0/autoresearch/configs/E003-recurrent-standard.json \
  --checkpoint artifacts/spider_v0/autoresearch/runs/E003-recurrent-standard/checkpoint.pt \
  --split validation_path_length_ood --cases 32
```

The binary checkpoints are reproducible local artifacts and are ignored by
Git; their configs, hashes, histories, metrics, and selection manifest are
tracked. Start with the [Spider final
report](docs/spider_v0/FINAL_REPORT.md), then see
[training](docs/spider_v0/TRAINING.md), [design](docs/spider_v0/DESIGN.md),
and the [dataset card](docs/spider_v0/DATASET_CARD.md).

## Spider v0.1 closed-loop alignment

Generate the v0.2 manifests without materialising sealed cases:

```bash
.venv/bin/python scripts/generate_spider_v0_2_dataset.py
```

Run the frozen ten-record controller-alignment matrix:

```bash
.venv/bin/python scripts/run_spider_v0_1_autoresearch.py \
  --steps 400 --train-cases 512 --device cuda --dtype float32
```

The one-time sealed command is intentionally guarded by finalist, config,
checkpoint, dataset, and split hashes. It must be run only after freezing the
finalist and threshold; a second invocation refuses access:

```bash
.venv/bin/python scripts/evaluate_spider_v0_1.py --allow-v0-2-sealed
```

The post-sealed long-run protocol uses three recurrent and three pooled runs
at 5,000 steps each on an H100 or A100 (A100 fallback only; no lower
accelerator fallback). Inspect the frozen protocol before launching:

```bash
python -m json.tool artifacts/spider_v0_1/COLAB_5K_PROTOCOL.json
```

The accepted recovery matrix uses
`scripts/colab_spider_v0_1_single_run.py`, one fresh A100 session per frozen
model/seed specification, and downloads each result before releasing its
session. Rebuild the verified six-run ledger and summary with:

```bash
.venv/bin/python scripts/aggregate_spider_v0_1_colab_runs.py
```

The post-sealed 5k diagnostic favored pooled (`0.3868`) over recurrent
(`0.3726`) on the registered primary metric. It cannot change the already
frozen finalist or sealed result. All six standalone checkpoints, complete
archives, and aggregate reports are backed up in the
[verified Google Drive folder](https://drive.google.com/drive/folders/10Pmjb0lBATNtGWyf823SB4qHAYaZ7Euw).
Interrupted T4, lost-session, and concurrent-session attempts remain recorded
and are excluded from all aggregates.

See the [Spider v0.1 final
report](docs/spider_v0_1/FINAL_REPORT.md), [failure
audit](docs/spider_v0_1/FAILURE_AUDIT.md), [training
protocol](docs/spider_v0_1/TRAINING.md), and [v0.2 dataset
card](docs/spider_v0_1/DATASET_CARD.md). Spider v0 remains immutable historical
evidence; v0.1 does not reinterpret its sealed result.

## Spider v0.2 recurrence-utility diagnostic

Generate the non-sealed, matched recurrence-necessity splits:

```bash
.venv/bin/python scripts/generate_spider_recurrence_dataset.py
```

Run a short fixed-horizon smoke test. Learned stopping is suppressed, while
frontier and evidence actions remain learned:

```bash
.venv/bin/python scripts/train_spider_recurrence.py \
  --config configs/spider_v0_2/recurrent_recurrence.json \
  --experiment-id smoke-recurrent \
  --output-dir /tmp/spider-v02-smoke \
  --steps 2 --train-cases 4 --eval-cases 4
```

The registered comparison is exactly three paired 6,000-step recurrent and
pooled runs. After every run archive and all 36 checkpoints are independently
verified and backed up, build the fail-closed aggregate and figures:

```bash
.venv/bin/python scripts/aggregate_spider_v0_2_training.py
.venv/bin/python scripts/render_spider_v0_2_training_plots.py
```

The aggregator requires all 42 registered Drive artifacts, the frozen source
and split hashes, and zero sealed access. Checkpoints and archives are stored
in the [Spider v0.2 Drive
folder](https://drive.google.com/drive/folders/1A8QnvZKDSWeiTXvi6RwYx76LFVAcDZRw).
See the [frozen design](docs/spider_v0_2/DESIGN.md), [training
protocol](docs/spider_v0_2/TRAINING.md), and [dataset
card](docs/spider_v0_2/DATASET_CARD.md). The completed three-seed result,
state-use interventions, limitations, and next recommendation are in the
[final report](docs/spider_v0_2/FINAL_REPORT.md); the certified machine-readable
aggregate is
[`artifacts/spider_v0_2/training/TRAINING_SUMMARY.json`](artifacts/spider_v0_2/training/TRAINING_SUMMARY.json).

## Spider v0.3 evidence and termination repair

Spider v0.3 preserves the completed v0.2 result and first traces every required
evidence item through reachability, enumeration, scoring, selection, and exact
recording. The preserved six-checkpoint diagnostic found 100% reachable and
scored-positive coverage; the measured bottleneck is evidence ranking and
selection, not candidate generation.

Regenerate the grouped, non-sealed development protocol and run a CPU smoke
evaluation with:

```bash
.venv/bin/python scripts/generate_spider_v0_3_dataset.py
.venv/bin/python scripts/spider_v0_3_evaluator.py \
  --config configs/spider_v0_3/evidence_smoke_cpu.json \
  --experiment-id v03-smoke \
  --output-dir /tmp/spider-v03-smoke \
  --stop-after-steps 2 \
  --train-cases 4 --calibration-cases 4 --evaluation-cases 4
```

The registered E0/E1/E2 evidence matrix is:

```bash
.venv/bin/python scripts/run_spider_v0_3_autoresearch.py \
  --phase all --output-root artifacts/spider_v0_3/evidence
```

See the [frozen design](docs/spider_v0_3/DESIGN.md), [preserved-checkpoint
diagnosis](docs/spider_v0_3/EVIDENCE_DIAGNOSTIC.md), and [durable A100/Drive
training protocol](docs/spider_v0_3/TRAINING.md). No Spider v0 or v0.2 sealed
data is opened by this work.
