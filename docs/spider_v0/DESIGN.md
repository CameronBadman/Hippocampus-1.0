# Spider v0 frozen design

Status: **implementation contract**

Frozen: 2026-07-26 after substrate checkpoint `2ad1861` and research checkpoint
`0975093`.

## Purpose

Spider v0 tests whether a small parameter-tied processor can execute reusable
graph traversals over exchangeable vector manifolds. It is a synthetic
structural benchmark, not a natural-language or production-memory system.

The implementation has three deliberately separated layers:

1. **Supervisor layer:** generates latent graph programs, exact answers,
   parallel traces, aligned views, and counterfactuals.
2. **Packed data layer:** renders observable atoms and packs them into the
   existing `PackedTopology` and `PackedGraph` structures.
3. **Execution layer:** scores packed frontier expansions, gathers packed
   manifolds, updates transient hypotheses/evidence, and executes a
   deterministic sparse controller.

Only observable manifolds, snapshot-local topology IDs, and exact budget state
cross into the execution layer. The program AST and oracle labels do not.

## Dependency direction

```text
program generator -> oracle/verifier -> renderer/batcher -> PackedGraph
                                                         |
                                                         v
                                            SparseWavefrontController
                                                         |
                                              PackedTopology expansion
                                                         |
                                           Spider/Baseline candidate scorer
```

The model never calls the generator. The controller never reconstructs Python
adjacency. All model-frontier traversal starts with
`PackedTopology.expand_frontier`.

## Package boundaries

```text
hippocampus/
  programs/
    schema.py            latent and observable dataclasses
    generator.py         four exact graph-program families
    oracle.py            parallel traces and mechanical verifier
    renderer.py          deterministic exchangeable vector renderer
    batching.py          PackedTopology/PackedGraph conversion and alignments
    splits.py            immutable split specifications and manifests
    counterfactuals.py   one-edit interventions
    equivalent_views.py aligned surface/topology transformations
    leakage.py           metadata-only diagnostics
  spider/
    config.py            model, controller, loss, and rollout configuration
    set_attention.py     padded position-free attention primitives
    multiset.py          explicit set-to-set reads
    arc_processor.py     standard/compositional candidate transitions
    context_refiner.py   bounded destination-context refinement
    evidence.py          neural evidence update and exact ledger records
    policy_heads.py      candidate policy/value outputs
    terminator.py        explicit continuation/answer/unknown head
    hypothesis.py        tensor hypothesis batches and trace references
    controller.py        deterministic sparse wavefront execution
    model.py             tied recurrent Spider and common scorer protocol
    baselines.py         pooled and flat-Transformer controls
    losses.py            opportunity-normalised multi-target losses
    training.py          staged training and checkpointing
    metrics.py           traversal/evidence/context/selective metrics
    evaluation.py        deterministic ID/OOD evaluation
```

Some small modules may be combined when separation would add ceremony, but the
dependency boundaries above remain.

## Program and observation schema

### Supervisor-only structures

`GraphProgramCase` contains:

- stable case ID and generation seed;
- a `ProgramSpec` whose family enum is supervisor-only;
- latent nodes and logical edges;
- observable query, node-summary, node-context, and edge atom collections;
- exact answer and evidence node sets;
- all declared valid traversal paths;
- a `ParallelOracleTrace`;
- answerability and exact termination target;
- aligned equivalent-view and optional counterfactual metadata.

`OracleRound` contains:

- active latent node occurrences;
- every acceptable latent edge expansion;
- optional priority tiers;
- per-candidate remaining distance/utility;
- context reads with positive value of information;
- acceptable evidence inclusions;
- the terminal target after the round.

The oracle trace is a set-valued supervision object. It does not prescribe an
order among equally legal expansions.

### Observable atoms

An `ObservableAtom` is an unordered symbolic/scalar observation:

- zero or more opaque surface symbols;
- at most one observable scalar value.

Symbols are re-keyed surface codes. Matching symbols are reused only when two
observations may legitimately be compared within a case. Scalars such as time,
confidence, or support polarity are values in observation rows; they are not
fixed named feature slots.

An atom has no answer flag, path flag, node ID, task enum, relation ID, row
position, or provenance feature.

### Packed alignment

`PackedProgramBatch` owns:

- a `PackedGraph`;
- rendered query manifolds as a `PackedManifoldFamily`;
- case-to-graph and latent-to-snapshot alignment tensors;
- oracle targets translated to snapshot-local node, edge, and arc IDs;
- supervisor metadata retained outside the model input.

Topology packing is reused across equivalent differentiable value snapshots
where topology is unchanged. Rendered fixed values are rebuilt only when a
surface view changes.

## Initial task families

### Direct and relational lookup

A query identifies a start observation and a desired observable value. Several
neighbours can match superficial attributes, but only edges whose observable
constraints are valid lead to the answer. Positive, absent, disconnected, and
missing-edge cases are generated.

### Reachability and shortest valid path

Directed paths compete through distractor branches and blocked edges. The
oracle contains all shortest valid wavefront expansions and exact
distance-to-go. Training paths are length 1--4; path-length OOD is 5--8.

### Latest-valid / supersession

Candidate record nodes expose coarse identity in summaries. Exact timestamp,
validity, or supersession observations are held in contexts. At least a
controlled subset of cases cannot be solved correctly without a context read.

### Corroboration and conflict

Independent graph branches contain supporting or contradicting observations.
An answer requires sufficient independent support; unresolved support and
contradiction terminates as `UNKNOWN_CONFLICT`. Multiple selected messages
update a query-global evidence manifold and exact ledger.

## Split freeze

The generator version, split seed ranges, case IDs, and hashes are frozen
before architecture tuning. Split manifests store generator configuration and
SHA-256 digests, not merely filenames.

| Split | Intended shift |
|---|---|
| `train` | 8--32 nodes, valid paths 1--4 |
| `validation_id` | held-out seeds under training ranges |
| `validation_graph_size_ood` | 64--128 nodes |
| `validation_path_length_ood` | valid paths 5--8 |
| `validation_topology_ood` | held-out motif/composition templates |
| `validation_cardinality_ood` | larger manifold row counts |
| `validation_equivalent_view_ood` | unseen surface codebook/domain |
| `validation_composition_ood` | held-out program compositions |
| `test_sealed` | aggregate held-out seeds; opened once |

The later 9--12-hop/256-node run is a stress report, not an architecture
selection metric.

## Deterministic renderer

The first renderer is frozen and non-learned:

1. Each opaque surface symbol maps to a seeded, normalised code vector.
2. A scalar atom combines its symbol code with a deterministic continuous
   scalar encoding.
3. A seeded family projection maps atoms to the corresponding
   `GraphSchema` width.
4. Row order is permuted by a supplied `torch.Generator`.

The same symbol in one view maps to the same code wherever legitimate matching
is required. Equivalent views use a different surface re-keying and can use a
different family projection seed while preserving latent alignments.

Summaries are an explicit subset/coarsening of contexts. Context-only decisive
atoms are recorded by the oracle verifier. Empty contexts remain legal.

An optional learnable family projection maps fixed rendered rows to `d_model`.
The underlying rendered values remain fixed during Spider v0 experiments.

## Common candidate-scorer protocol

Every model implements a common tensor interface:

```text
score_candidates(
    packed_batch,
    query_batch,
    hypotheses,
    frontier_expansion,
    controller_state,
) -> CandidateOutputs
```

`CandidateOutputs` contains:

- proposed next path manifolds `[candidate, path_rows, d_model]`;
- priority logits;
- expand/retain logits;
- context-read value/logits;
- evidence-inclusion logits;
- remaining cost/utility;
- support and conflict logits.

The context refiner accepts only candidates selected by the exact context-read
budget and returns updated states and heads for those candidate indices.

## Position-free attention

The correctness backend uses padded PyTorch scaled dot-product attention:

- no positional embeddings;
- no causal mask;
- boolean validity masks only;
- all-empty key/value sets bypass attention and produce a defined zero update;
- FP32 is the reference;
- CUDA BF16 is enabled only when the existing pack configuration and device
  support it.

The data layer's power-of-two or single layouts are used to construct padded
batches. CUDA varlen is an explicitly reported optional backend and is disabled
unless the installed PyTorch API, dtype/head dimensions, and equivalence tests
all pass. Spider v0 never silently changes backend.

## Recurrent multi-set Spider

Default model dimensions are configurable and initially:

- `d_model=128`;
- 4 attention heads;
- 2 tied Spider blocks;
- 4--8 recurrent controller rounds;
- 8 path-state rows;
- 8 global-evidence rows;
- 4 low-rank compositional transforms;
- 2 hypotheses per destination;
- global frontier width 32;
- dropout 0.

A lightweight tiny-overfit configuration uses the same interfaces with smaller
dimensions.

For each candidate arc, a tied block performs:

1. path reads query;
2. path reads source summary;
3. path reads edge;
4. path reads destination summary;
5. optional edge-conditioned value composition;
6. path self-attention;
7. identity-biased gated residual update.

Rows are pooled only after these updates for policy/value heads.

### Standard arc processor

The standard control uses ordinary cross-attention for the edge and
destination reads. The edge residual is added through a projected,
symmetrically pooled representation.

### Compositional arc processor

Content attention selects destination rows. A symmetric edge representation
produces mixture weights over a small shared low-rank value-transform bank.
The mixed transformed destination update and direct edge residual are added to
the path update.

No transform has a named meaning, and complete Transformer parameters are not
generated per edge.

## Transient hypotheses

`HypothesisBatch` is a tensor structure containing:

- snapshot-local current node IDs;
- path-state manifolds;
- accumulated scores;
- depths;
- parent hypothesis references;
- incoming arc and logical-edge references;
- context-read flags;
- graph IDs and controller budget state.

Trace references point to immutable ledger entries; Python path lists are not
copied through the candidate hot path. Persistent node manifolds are never
modified by hypothesis updates.

The same node can own multiple hypotheses. A stable vectorised ranking enforces
the per-destination cap before the global frontier cap.

## Global evidence

Each graph/query owns a small homogeneous evidence manifold initialised from
learned rows conditioned on the query. Only candidates selected as evidence
can update it.

Alongside the neural state, `EvidenceLedgerEntry` records exact:

- graph, node, edge, and arc IDs;
- round and source frontier occurrence;
- parent hypothesis;
- whether context was read.

The ledger is audit state, not a neural input.

## Deterministic sparse controller

For every recurrent round:

1. Call `PackedTopology.expand_frontier` with all current node occurrences.
2. Use `frontier_positions` to repeat path hypotheses without a per-node loop.
3. Gather source summaries, edge manifolds, and destination summaries from the
   packed families.
4. Score all candidate arcs.
5. Stably rank by descending priority, preserving earlier stable keys and
   using snapshot-local arc ID as the final tie-break.
6. Apply the per-destination hypothesis cap tensorially.
7. Apply the global frontier cap.
8. Select bounded context reads, refine those candidates, and rerank if the
   configured policy permits it.
9. Append selected exact evidence records.
10. Execute the explicit termination decision or continue.

Duplicate frontier occurrences remain distinct because candidate ownership is
indexed through `frontier_positions`. Isolated occurrences contribute repeated
`arc_offsets` and no candidates.

Supported rollout modes are:

- `oracle`: construct the next hypothesis set from every acceptable oracle
  action;
- `model`: use deterministic model ranking;
- `mixed`: choose the mode per case/round from a seeded schedule.

No discrete controller selection is differentiated through.

## Termination and abstention

The terminator consumes symmetrically pooled query/evidence/frontier state,
candidate support/conflict statistics, and exact controller flags:

- frontier exhausted;
- search budget exhausted;
- context budget exhausted;
- remaining context reads;
- predicted remaining utility.

Its labels are:

1. `CONTINUE`;
2. `ANSWER`;
3. `UNKNOWN_ABSENT`;
4. `UNKNOWN_CONFLICT`;
5. `UNKNOWN_INCOMPLETE`;
6. `UNKNOWN_UNSUPPORTED`.

Unknown is never implemented as a maximum-score threshold.

## Baselines

### Pooled scorer

Mean and max pool every input manifold, concatenate the operational family
representations, and score with an MLP. It has no recurrent set processing.

### Flat Transformer

Project all family rows, add only operational family boundary embeddings,
concatenate, and apply position-free self-attention with padding masks. There
are no row positions and no compositional edge mechanism.

### Recurrent standard Spider

The tied multi-set processor with standard edge/destination cross-attention.

### Recurrent compositional Spider

The recurrent standard Spider plus the optional edge-conditioned transform
bank.

Comparisons record parameter counts and keep configurations within the
pre-registered parameter range.

## Loss contract

Every loss returns raw value, configured weighted value, and opportunity count.
Zero-opportunity losses return differentiable zero and count zero.

Candidate losses:

- multi-positive listwise priority;
- expand/retain binary classification;
- context-read value;
- evidence inclusion;
- support/conflict;
- remaining cost/utility.

Global losses:

- termination/unknown reason;
- evidence-set selection;
- equivalent-view behavioural consistency;
- row-permutation behavioural consistency;
- sparse expansion/context-read cost.

Losses are normalised by valid opportunities. A canonical edge is never used
when multiple actions are acceptable.

## Training stages

1. **Tiny overfit:** fixed 48-case dataset, oracle frontiers, all supervised
   heads, deterministic evaluation.
2. **Oracle-frontier structural:** generated training cases with exact
   wavefronts.
3. **Mixed rollout:** scheduled oracle fractions 1.0, 0.75, 0.5, then 0.25;
   encountered invalid candidates become hard negatives.

Reinforcement learning is out of scope.

## Invariance and determinism contracts

- Seeded row shuffling leaves model decisions unchanged within configured
  tolerance.
- Repeated owner selections preserve duplicate gradient accumulation.
- Repeated deterministic model rollout yields identical selected arc traces.
- Model parameters contain no positional-embedding table.
- Recurrent rounds reference identical block objects.
- Equivalent views align decisions through latent mappings, not vector
  coordinate equality.

## Allowed design changes after freeze

This document changes only if:

- a test demonstrates a correctness problem;
- an implemented packed-substrate API contradicts an assumption;
- a controlled experiment provides concrete evidence for a revision.

Every revision must record the motivating test or experiment ID.
