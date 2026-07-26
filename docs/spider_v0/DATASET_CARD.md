# Spider v0 synthetic graph-program dataset card

Version: `spider-programs-v0.1`

Status: generator contract frozen; manifests and measured statistics are added
after generation.

## Intended use

This synthetic dataset isolates sparse graph traversal, evidence collection,
context-read decisions, abstention, and size/path generalisation. It is
intended for structural model development only.

It does not measure natural-language understanding, factual memory, safety,
real-world reasoning, or production readiness.

## Unit of data

One `GraphProgramCase` contains a latent exact program instance plus one or
more observable views. Model-visible data consists only of:

- graph topology;
- unordered query rows;
- unordered node summary rows;
- unordered node context rows;
- unordered logical-edge rows;
- exact controller budget state during execution.

Supervisor-only data includes the program family, latent identities, answers,
paths, parallel oracle trace, context value-of-information, and intervention
metadata.

## Program families

| Family | Core requirement | Main negative cases |
|---|---|---|
| lookup | select a structurally valid one-hop match | absent, blocked edge, plausible distractor |
| reachability | follow one of all shortest valid directed paths | disconnected, missing/reversed edge, exhausted budget |
| latest-valid | compare context-only validity/time observations | stale, invalid, ambiguous latest, context budget exhausted |
| corroboration | collect independent support and detect contradiction | absent support, unresolved conflict, insufficient search |

Families are balanced within each ordinary split. Split generators also
counterbalance answerability and unknown reason where the requested graph
ranges permit it.

## Frozen split specification

| Split | Cases in initial generated artifact | Nodes | Valid path length | View/cardinality policy |
|---|---:|---:|---:|---|
| train | 512 | 8--32 | 1--4 | training domains/cardinalities |
| validation_id | 128 | 8--32 | 1--4 | held-out seeds |
| validation_graph_size_ood | 96 | 64--128 | 1--8 | training domains |
| validation_path_length_ood | 96 | 16--64 | 5--8 | training domains |
| validation_topology_ood | 96 | 16--64 | 2--8 | held-out motifs |
| validation_cardinality_ood | 96 | 8--32 | 1--4 | 2--4x row counts |
| validation_equivalent_view_ood | 96 | 8--32 | 1--4 | held-out codebook/domain |
| validation_composition_ood | 96 | 16--64 | 2--8 | held-out family composition |
| test_sealed | 256 | mixed | mixed | aggregate held-out seeds |

The tiny-overfit fixture is a separate deterministic 48-case set. It is not
part of model selection metrics.

The initial experiment runner may use a deterministic reduced case count per
validation split to fit the pre-registered compute budget. Manifests always
record both generated and evaluated case IDs.

## Generation and randomisation

Independent seeded streams randomise:

- opaque node surface symbols;
- node insertion order;
- logical-edge insertion order;
- answer and start locations;
- degree and distractor count;
- component count;
- valid path depth;
- positive/negative outcome;
- manifold row count and row order;
- surface codebook/domain;
- query and relation renderings.

Graph node indices and program enums never become neural features.

## Observable renderer

Rows are deterministic vectorisations of unordered observable atoms. Opaque
symbols use seeded normalised codes. A scalar is encoded continuously in its
own atom row. Rows receive no local index, positional embedding, answer flag,
path flag, provenance feature, or semantic-slot label.

Node summaries are deliberately lossy. Contexts can contain decisive
timestamp, validity, or conflict observations not present in summaries.

Equivalent views preserve latent truth and alignment while changing surface
symbols, topology insertion order, irrelevant observations, distractors, and
row permutations.

## Counterfactuals

Supported one-edit interventions are:

- remove a decisive edge;
- reverse a decisive edge;
- change a temporal value;
- add a newer conflict;
- invalidate one source;
- replace one endpoint;
- disconnect the only valid path.

Every pair stores the exact changed latent object and expected affected labels.
The verifier rejects pairs with unrecorded observable or target differences.

## Quality controls

Before serialisation, the mechanical verifier checks:

- every oracle expansion resolves to a valid directed arc;
- all paths respect direction and edge validity;
- evidence nodes satisfy the program;
- answerability and unknown reason match exact execution;
- declared context reads reveal decisive context-only information;
- equivalent views preserve truth and align latent objects;
- counterfactual labels match an independently recomputed oracle.

Dataset tests additionally check generator determinism, split disjointness,
row exchangeability, variable cardinality, and summary/context distinction.

## Leakage diagnostics

Metadata-only baselines are fitted to:

- graph size;
- edge count;
- degree statistics;
- start-node degree;
- manifold cardinalities;
- distractor count;
- component count;
- insertion positions;
- configured budget.

Diagnostics report task-family and answerability prediction against the
majority/chance baseline. A split artifact is rejected if a simple metadata
classifier exceeds the pre-registered tolerance by more than 0.10 absolute on
answerability or reveals a fixed answer/edge position.

The diagnostic is a guard against obvious leakage, not proof that the
benchmark has no shortcuts.

The generated manifest diagnostic used 128 balanced cases:

- majority answerability accuracy: 0.5000;
- best metadata decision-stump accuracy: 0.4844;
- measured answerability advantage: 0.0000;
- most-common answer position rate: 0.1094;
- most-common decisive-edge position rate: 0.1076;
- oracle verification errors: 0.

The aggregate split-manifest SHA-256 is
`3f93841b41f025e72e176be4b0934b18a9ab1b8c37e5449cc971abb9684c8404`.
Exact per-split digests are stored in
`artifacts/spider_v0/splits/MANIFEST_INDEX.json`.

## Sealed-test policy

The sealed manifest is generated and hashed with the other splits but its
examples are not loaded by architecture-search commands. The selected
configuration and finalist seeds are frozen first. The aggregate sealed test
is evaluated once, with the invocation and resulting digest appended to the
experiment ledger.

## Known limitations

- Synthetic observable codes are far simpler than language.
- Exact generators can underrepresent ambiguity found in real data.
- Re-keyed codebooks test a narrow form of view shift.
- The four task families do not establish a general-purpose interpreter.
- Metadata diagnostics cannot detect every neural shortcut.
- Successful size/path extrapolation would be benchmark evidence only.
- The initial AutoResearch source commit predates the added explicit
  incomplete/unsupported negative variants. Its source commit and split digest
  are retained in every experiment record; the variants are covered by the
  final generator/tests and tiny-overfit gate rather than retroactively folded
  into those experiment results.
