# Spider v0 research decisions

Status: **frozen for the initial implementation**

Decision date: 2026-07-26

Substrate checkpoint: `2ad1861`

## Scope and evidence labels

This brief freezes the first Spider v0 architecture before model tuning. It is
not a claim that every cited mechanism transfers to Hippocampus.

- **Established reuse** means the implementation directly adopts a published
  mechanism or a conservative engineering consequence of it.
- **Hippocampus inference** means the mechanism motivates a design choice, but
  the exact choice has not been established by the cited work.
- **Experimental hypothesis** means the milestone must compare the choice
  against a simpler control and preserve a negative result.

The generator AST, task family, oracle trace, targets, and split bookkeeping
remain supervisor-only. None are model inputs.

## Frozen decision summary

| Topic | Spider v0 decision | Evidence status |
|---|---|---|
| Recurrent processing | Two tied multi-set blocks, reused for every round | Established reuse plus Hippocampus inference |
| Search | Exact CSR expansion followed by learned scoring and deterministic sparse top-k | Established reuse plus Hippocampus inference |
| Set processing | Position-free self- and cross-attention over separately identified operational families | Established reuse |
| Parallel supervision | Acceptable action sets and wavefront rounds, never one arbitrary path order | Established reuse |
| Edge interpretation | Standard edge cross-attention is the control; compositional modulation is optional | Experimental hypothesis |
| Path multiplicity | Bounded hypotheses per destination, compared with one-hypothesis pruning | Experimental hypothesis |
| Evidence | Small neural evidence set plus an exact non-neural audit ledger | Hippocampus inference |
| Context reads | Supervised value-of-information under an exact read budget | Hippocampus inference |
| Abstention | Explicit answer/unknown-reason head trained jointly | Established reuse plus Hippocampus inference |
| Equivalent views | Behavioural consistency, not coordinate equality | Established reuse plus Hippocampus inference |
| Edge swapping | Optional aligned-view augmentation and consistency loss | Experimental hypothesis |
| Verification | Exact generator/oracle verifier; no learned verifier in v0 | Conservative established reuse |

## Neural Bellman-Ford Networks

Primary source: Zhu et al., [Neural Bellman-Ford Networks: A General Graph
Neural Network Framework for Link
Prediction](https://papers.neurips.cc/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html),
NeurIPS 2021.

- **Relevant mechanism:** NBFNet expresses path reasoning as learned
  generalisations of Bellman-Ford's indicator, message, and aggregation
  operators. A shared transition composes information along paths while an
  aggregation combines alternatives.
- **Problem addressed:** multi-hop relational reasoning without enumerating a
  fixed catalogue of path templates.
- **Decision:** implement now at the architectural level: repeated shared
  transition processing, explicit frontier expansion, and separate candidate
  message scoring. Do not copy NBFNet's relation-ID embedding interface.
- **Concrete consequence:** one Spider transition consumes the source
  hypothesis, source and destination manifolds, edge manifold, query, and
  global evidence. The controller performs the graph recurrence; the neural
  processor does not own topology.
- **Risks and shortcuts:** dense propagation can learn degree or reachability
  shortcuts, and relation embeddings would violate the milestone's
  representation rules. Sparse pruning can also remove the only valid path.
- **Evidence label:** tied graph recurrence is established reuse. Applying it
  to exchangeable edge manifolds and transient path-state manifolds is a
  Hippocampus-specific inference.

## A*Net

Primary source: Zhu et al., [A*Net: A Scalable Path-based Reasoning Approach
for Knowledge
Graphs](https://papers.neurips.cc/paper_files/paper/2023/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html),
NeurIPS 2023.

- **Relevant mechanism:** learned priority scores select a small set of nodes
  and edges from a much larger path search, analogous to an A*-style heuristic.
- **Problem addressed:** the cost of propagating over every possible path in a
  large graph.
- **Decision:** implement now as a deterministic sparse wavefront controller
  around the neural scorer. The controller, rather than the network, owns
  budgets, stable sorting, hypothesis caps, and termination execution.
- **Concrete consequence:** `PackedTopology.expand_frontier` produces the
  complete candidate arc tensor; neural heads propose priority and utility;
  stable tensor operations select the bounded next frontier.
- **Risks and shortcuts:** a learned heuristic may exploit degree, graph size,
  or answer-location correlations. Early pruning creates irreversible rollout
  errors. Oracle-frontier training and explicit invalid-expansion/search-cost
  metrics are therefore required before model rollouts.
- **Evidence label:** learned sparse path prioritisation is established reuse.
  The exact deterministic controller and snapshot-local arc-ID tie-break are
  Hippocampus-specific engineering.

## Set Transformer

Primary source: Lee et al., [Set Transformer: A Framework for
Attention-based Permutation-Invariant Neural
Networks](https://proceedings.mlr.press/v97/lee19d.html), ICML 2019.

- **Relevant mechanism:** attention blocks process sets without positional
  encodings, with equivariant intermediate representations and invariant
  pooling only where a set-level output is required.
- **Problem addressed:** modelling interactions among exchangeable elements
  without imposing an arbitrary order.
- **Decision:** implement now. Every manifold attention operation is
  position-free, non-causal, and masked only for padding validity. Pooling is
  confined to policy/value heads.
- **Concrete consequence:** row permutation must permute row-level outputs and
  leave controller decisions invariant within numerical tolerance.
- **Risks and shortcuts:** padding masks can accidentally become cardinality
  cues; learned seed vectors can become named slots if exposed as semantic
  roles. Spider's path/evidence slots are homogeneous learned initial rows and
  are never assigned human meanings.
- **Evidence label:** position-free set attention is established reuse.

## Multi-Set Transformer

Primary source: Selby et al., [Multi-Set Transformer: A Framework for
Attention-based Permutation-Invariant Neural Networks for Multiple
Sets](https://proceedings.mlr.press/v180/selby22a.html), ICML 2022.

- **Relevant mechanism:** separate self- and cross-attention blocks model
  interactions within and between multiple sets while preserving the required
  permutation symmetries.
- **Problem addressed:** flattening distinct sets obscures their interaction
  structure and can make the model learn brittle boundary conventions.
- **Decision:** implement now as the main Spider processor. Query, path,
  summaries, edge, context, and evidence remain distinct operational
  collections.
- **Concrete consequence:** the baseline includes a flat position-free
  Transformer, while the Spider performs explicit set-to-set reads in a frozen
  order: query, source summary, edge, destination summary, optional context,
  then path self-attention.
- **Risks and shortcuts:** operational family projections can still encode
  family identity, which is permitted, but must not encode program family,
  node type, or relation type. Cross-attention order is an architectural prior
  and must be reported as such.
- **Evidence label:** multi-set attention is established reuse; the selected
  read sequence is a Hippocampus-specific inference.

## Compositional Attention

Primary source: Mittal et al., [Compositional
Attention](https://openreview.net/forum?id=IwJPj2MBcIa), ICLR 2022.

- **Relevant mechanism:** content search and value retrieval are separated so
  query-key matches can be recombined with different value transformations.
- **Problem addressed:** ordinary attention entangles where to retrieve
  information with how retrieved information is transformed.
- **Decision:** implement as an optional Spider component, with standard
  cross-attention as the required control. A small shared bank of low-rank
  value transforms is mixed continuously from the edge manifold. A direct
  residual edge path remains available.
- **Concrete consequence:** the experiment ledger must compare otherwise
  matched standard and compositional arc processors.
- **Risks and shortcuts:** the transform bank can collapse to one component,
  become an implicit finite relation catalogue, or overfit surface codebooks.
  Mixture entropy and component use are diagnostics only, not semantic labels.
- **Evidence label:** separation of search and retrieval is established.
  **Edge-conditioned compositional attention is a genuinely uncertain
  experimental hypothesis.**

## Relational attention and Dual Attention

Primary sources:

- Diao and Loynd, [Relational Attention: Generalizing Transformers for Graph
  Processing](https://openreview.net/forum?id=cFuMmbWiN6), ICLR 2023.
- Altabaa et al., [Disentangling and Integrating Relational and Sensory
  Information in Transformer
  Architectures](https://arxiv.org/abs/2405.16727), 2024.

- **Relevant mechanism:** relational attention treats pairwise relations as
  first-class representations; Dual Attention separates sensory/content
  processing from relational processing and then integrates them.
- **Problem addressed:** standard self-attention primarily updates object
  representations and can underrepresent evolving relational state.
- **Decision:** use as design motivation, not a full implementation. Spider
  keeps edge manifolds explicit and separates content search from relation
  interpretation, but does not maintain a dense learned relation tensor or
  generate Transformer weights per edge.
- **Concrete consequence:** the edge manifold is consumed on every candidate
  transition and is retained through a direct residual route even when the
  compositional adapter is enabled.
- **Risks and shortcuts:** dense pairwise relation state would defeat sparse
  graph execution; surface edge codes can act like forbidden relation IDs if
  the renderer is not re-keyed and counterbalanced.
- **Evidence label:** explicit relation processing is established reuse.
  Treating a variable edge manifold as a relational program is a
  Hippocampus-specific inference.

## Looped and recurrent Transformers for graph algorithms

Primary sources:

- de Luca and Fountoulakis, [Simulation of Graph Algorithms with Looped
  Transformers](https://arxiv.org/abs/2402.01107), 2024.
- Yang et al., [Looped Transformers are Better at Learning Learning
  Algorithms](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html),
  ICLR 2024.

- **Relevant mechanism:** one parameter-tied block is applied repeatedly, so
  computation depth can vary independently of parameter count. The graph
  simulation result shows that looped attention architectures can express
  several classical algorithms under explicit representation assumptions.
- **Problem addressed:** algorithm length and graph size should not require a
  different learned layer at every step.
- **Decision:** implement tied recurrent processing as the default and an
  untied-per-round ablation. Train on 4--8 rounds and evaluate longer rollouts
  without absolute hop embeddings.
- **Concrete consequence:** all recurrent rounds call the same Spider block
  objects; a test checks parameter identity, not merely equal initial values.
- **Risks and shortcuts:** theoretical simulation does not establish
  learnability or finite-precision OOD generalisation. Repeated residual
  updates can drift or oversmooth. Identity-biased residual gates and round
  diagnostics are included.
- **Evidence label:** parameter tying is established reuse. Extrapolation for
  this manifold representation remains an empirical question.

## Parallel and asynchronous neural algorithmic reasoning

Primary sources:

- Dudzik et al., [Parallel Algorithms Align with Neural
  Execution](https://openreview.net/pdf?id=IC6kpv87LB), 2023.
- Rodionov and Prokhorenkova, [Asynchronous Algorithmic Alignment with
  Cocycles](https://openreview.net/forum?id=JV3fgFvL2J), 2025.

- **Relevant mechanism:** parallel algorithm supervision better matches
  synchronous neural message passing, while asynchronous formulations
  separate when an update is legal from the local state transition itself.
- **Problem addressed:** supervising one arbitrary sequential trace penalises
  other valid schedules and teaches incidental ordering.
- **Decision:** implement parallel oracle rounds and acceptable action sets
  now. Do not implement asynchronous event training in v0; preserve the trace
  schema so alternative legal schedules can be added later.
- **Concrete consequence:** candidate losses are multi-label/listwise over all
  acceptable expansions in a round. The verifier accepts every declared legal
  path rather than privileging a canonical next edge.
- **Risks and shortcuts:** a union of acceptable actions can include mutually
  incompatible choices if the oracle is careless. The trace verifier checks
  round legality and evidence consistency mechanically.
- **Evidence label:** parallel supervision is established reuse; the exact
  wavefront schema is Hippocampus-specific.

## Hint-ReLIC and causal regularisation

Primary source: Bevilacqua et al., [Neural Algorithmic Reasoning with Causal
Regularisation](https://proceedings.mlr.press/v202/bevilacqua23a.html), ICML
2023.

- **Relevant mechanism:** causally related augmentations are constrained to
  produce consistent intermediate algorithmic behaviour, reducing reliance
  on spurious input correlations.
- **Problem addressed:** neural algorithmic reasoners often fit training sizes
  while failing under distribution shifts.
- **Decision:** implement equivalent-view and row-permutation behavioural
  consistency. Compare logits/actions/evidence after latent alignment; do not
  require hidden manifold coordinates to be equal.
- **Concrete consequence:** split generation provides aligned views and
  counterfactual pairs. Consistency losses are separately logged and can be
  disabled.
- **Risks and shortcuts:** an invalid augmentation can change the correct
  trajectory; excessive consistency can collapse useful view-specific
  evidence. The oracle verifier checks truth preservation before a pair enters
  training.
- **Evidence label:** causal consistency is established reuse. The selected
  graph-view transformations and behavioural alignment are Hippocampus
  inferences.

## SelectiveNet and selective prediction

Primary source: Geifman and El-Yaniv, [SelectiveNet: A Deep Neural Network
with an Integrated Reject
Option](https://proceedings.mlr.press/v97/geifman19a.html), ICML 2019.

- **Relevant mechanism:** prediction and rejection are trained jointly under a
  coverage/risk objective rather than deriving abstention from a post-hoc
  maximum probability threshold.
- **Problem addressed:** a model should decline unsupported decisions while
  controlling how often it answers.
- **Decision:** implement an explicit six-way termination head from the first
  training stage: continue, answer, absent, conflict, incomplete, and
  unsupported. Report risk against coverage; do not claim softmax calibration.
- **Concrete consequence:** exact frontier exhaustion, search-budget state,
  and context-budget state are legitimate controller inputs to the terminator.
  Oracle answerability and unknown reason are targets, not model features.
- **Risks and shortcuts:** class imbalance can make always-unknown attractive;
  graph size or budget can reveal the label if generation is not
  counterbalanced. Coverage and false-answer/false-unknown rates are mandatory.
- **Evidence label:** joint reject-option training is established reuse. The
  reason-specific termination interface is a Hippocampus inference.

## Verifier-driven and trace-supervised reasoning

Primary sources:

- Mirman et al., [Training Neural Machines with Trace-Based
  Supervision](https://proceedings.mlr.press/v80/mirman18a.html), ICML 2018.
- Yu et al., [Scaling Flaws of Verifier-Guided Search in Mathematical
  Reasoning](https://arxiv.org/abs/2502.00271), 2025.

- **Relevant mechanism:** intermediate execution traces provide denser,
  mechanically checkable supervision. Verifier-guided search can focus
  computation, but verifier errors can prune valid solutions.
- **Problem addressed:** final-answer supervision alone does not identify
  which transitions, reads, or evidence selections were valid.
- **Decision:** implement an exact symbolic trace verifier and dense
  candidate/action supervision. Defer learned verifiers and
  reinforcement-learning search.
- **Concrete consequence:** every generated case is rejected unless its arcs,
  paths, context-read value, evidence, answerability, and counterfactual labels
  verify. Model mistakes encountered during mixed rollouts become supervised
  hard negatives.
- **Risks and shortcuts:** a learned verifier would introduce a second source
  of OOD error and could reward convincing invalid traces. An exact verifier
  can still encode a faulty benchmark specification, so tests include
  independent intervention checks.
- **Evidence label:** trace supervision and exact checking are established
  conservative practice. Applying the verifier to parallel graph-program
  rounds is Hippocampus-specific.

## Explicit experimental hypotheses

These are pre-registered as uncertain and cannot be presented as established
results.

### H1: edge-conditioned compositional attention

The edge manifold may productively choose how destination content transforms a
path state, beyond ordinary cross-attention. Compare:

1. standard edge and destination cross-attention;
2. the same processor plus a continuously mixed shared transform bank.

Primary outcomes are path-length OOD valid-frontier recall and
equivalent-view OOD decision accuracy. Component collapse and added parameter
count are reported.

### H2: functional edge-manifold swapping

Aligned edge manifolds with the same latent function may be interchangeable
across equivalent views even when their coordinates differ. Compare:

1. no swap training;
2. swap augmentation;
3. swap behavioural consistency;
4. cross-view swap evaluation without swap training.

This may improve functional abstraction, do nothing, or cause representational
collapse. Only behavioural correctness is required.

### H3: bounded multiple hypotheses per destination

Keeping two to four path hypotheses per node may preserve conflicting or
path-dependent evidence that one-state message passing loses. Compare matched
controllers with one and multiple hypotheses per destination. Report frontier
cost, duplicate rate, conflict accuracy, and OOD path validity. More
hypotheses are not assumed to be better.

## Deferred mechanisms

- Learned natural-language encoders and manifold writers.
- Learned manifold cardinality.
- Dense relational state over every node pair.
- Fully edge-generated Transformer weights.
- Learned verifier or reinforcement learning.
- Custom CUDA/Triton attention.
- Asynchronous event scheduling.
- Calibration beyond structural risk/coverage reporting.

These deferrals keep Spider v0 focused on whether exchangeable manifold
processing and sparse tied recurrence work at all.
