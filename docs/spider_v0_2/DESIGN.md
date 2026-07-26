# Spider v0.2: Recurrence-Utility Diagnostic

Status: frozen before implementation and evaluation.

Spider v0.2 asks one narrow question:

> Does the existing tied recurrent multi-set Spider use path state to improve
> multi-transition graph execution when learned termination cannot stop the
> rollout early?

This is a post-sealed diagnostic. It does not alter the historical Spider v0
or v0.1 conclusions, selection decisions, reports, checkpoints, or sealed-test
records. No Spider sealed split is an allowed input.

## Fixed-horizon execution

The controller continues to use the shared packed execution path:

1. `PackedTopology.expand_frontier`;
2. model candidate scoring;
3. learned context selection;
4. context refinement;
5. learned evidence selection and exact-ledger update;
6. learned frontier selection;
7. next-hypothesis construction.

Learned stopping is ignored until the configured horizon. The primary
diagnostic uses the exact number of oracle rounds for each case. Secondary
diagnostics use fixed horizons of four, six, and eight rounds. An empty
frontier remains empty and executes canonical no-candidate transitions until
the horizon; it is not silently replaced with an oracle frontier.

At the final round only, the existing termination head is evaluated. A final
`CONTINUE` becomes `UNKNOWN_INCOMPLETE` because the diagnostic budget has
ended. Intermediate termination logits are not consulted.

Two outcomes are reported separately:

- **structural success**: exact evidence-set recovery plus a semantically valid
  trace; for answerable cases the trace must include an oracle-valid route;
- **final autonomous success**: the existing answer/unknown decision contract,
  evaluated only after the horizon.

The architectural comparison is based on structural success. Final autonomous
success remains a diagnostic of the old termination head.

## State-use interventions

Interventions are applied before candidate scoring in rounds after round zero.
They never alter node IDs, graph IDs, controller budgets, exact ledgers, or the
packed graph.

- `none`: preserve the selected previous path state.
- `reset`: rebuild the query-conditioned initial path state for every active
  hypothesis.
- `detach`: preserve values but sever the autograd connection between rounds.
  Checkpoint inference should therefore be numerically identical; its purpose
  is a training-gradient ablation.
- `shuffle`: deterministically permute path states between hypotheses within
  each graph while leaving hypothesis topology and metadata fixed.
- `pooled_current_node`: replace prior state with a symmetric mean of the
  current node summary, repeated across path rows.

The intervention seed and round index determine shuffle order. No intervention
uses latent program state or oracle labels.

## Recurrence-necessity development split

The new development-only dataset version is
`spider-programs-v0.3-recurrence-dev`. It has no sealed split.

Each matched pair contains several equal-length branches of four to eight
transitions. First-hop neighbours have identical node summaries, degrees, and
manifold cardinalities. Intermediate nodes have matched local observations and
topology profiles. A branch token is observed on the first transition and a
comparison token is observed only on the final transition. Correct evidence
requires composing those observations across the path. Swapping first-hop
tokens between paired cases changes the correct branch while preserving the
multiset of local observations.

All branches remain legal search actions until the final comparison, so the
controller must retain multiple hypotheses. Early stopping is always
incorrect. The split is used for development and architectural diagnosis only.

## Pre-registered comparisons

1. Evaluate the three preserved recurrent and three preserved pooled v0.1
   checkpoints under learned, oracle-required, and fixed 4/6/8 horizons on all
   non-sealed v0.2 validation splits.
2. Evaluate each recurrent checkpoint under all five state interventions.
3. Train matched recurrent and pooled models on the recurrence-necessity train
   split with the same parameter range, controller, renderer, optimiser,
   schedule, cases, and training budget.
4. Repeat the fixed-horizon and state-use comparisons on the held-out
   recurrence-necessity validation split.

The recurrence is considered to show material state use only if either reset
or shuffle reduces structural success by at least 0.05 absolute on average and
does so in at least two of three seeds. The recurrent processor is considered
to earn its extra complexity only if its mean structural success exceeds the
pooled control by at least 0.02 and it wins at least two of three paired seeds.
All other outcomes are reported as inconclusive or negative.

## Post-frozen controller follow-ups

The following mechanisms are implemented behind explicit configuration and
are excluded from the fixed-horizon recurrent-versus-pooled comparison:

- `termination_mode="factorized"` predicts `evidence_sufficient`,
  `useful_work_remaining`, `answer_supported`, and `unknown_reason`.
  The controller permits STOP only when evidence is sufficient, the model
  predicts no useful work, the frontier is empty, or an exact execution budget
  is exhausted. ANSWER additionally requires both evidence sufficiency and
  answer support.
- `expansion_policy="learned_null"` adds a per-graph null-expansion logit.
  Selecting it suppresses all next-frontier candidates for that graph without
  itself executing global termination. Candidate expansion, evidence
  inclusion, and termination remain separate actions.

The factorised heads receive their own masked objectives. The null action is
supervised positive only when no depth-eligible candidate preserves a valid
completion. Existing flat/hierarchical terminators and threshold expansion
remain the defaults, so historical checkpoints load without new parameters.

Neither mechanism is used to rewrite or select the frozen fixed-horizon result.
