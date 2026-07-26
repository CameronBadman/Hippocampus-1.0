# Spider v0.1 failure audit

## Scope and method

This audit characterises the immutable Spider v0 implementation at commit
`8c1d74914f32a130311dc9eed08b9c21faaee7ae`. Source locations below refer to
that commit, before the v0.1 corrections. The baseline gate was:

```text
106 CPU tests passed
6 CUDA tests passed
NVIDIA GeForce RTX 5070 Ti, driver 595.84
```

Each item was verified by following the executable call path, not inferred
from the v0 final report. The named regression tests are introduced by v0.1;
they fail against the audited implementation and pass against the shared
transition.

## Findings

### A01 — candidate controller features differ

**Confirmed.** `oracle_rollout` and `mixed_rollout` call
`model.score_candidates` without controller features
(`training.py:599-605`, `training.py:741-747`). `SpiderModel` consequently
constructs an all-zero control tensor (`model.py:362-365`). Runtime constructs
six non-zero, budget-aware features and passes them explicitly
(`controller.py:209-253`, `controller.py:326-340`).

**Expected effect.** The recurrent processor is trained on one control
distribution and evaluated on another. Budget and depth coefficients are
untrained at runtime.

**Executable regression.**
`test_candidate_controls_are_identical_in_training_and_runtime`.

**Correction.** A pure `candidate_control_features` function is called only by
the shared proposal path used in every execution mode.

### A02 — mixed context reads are oracle actions

**Confirmed.** Mixed rollout selects every
`supervision.context_has_value` candidate and refines it before any scheduling
decision (`training.py:748-759`). `oracle_fraction` controls frontier selection
only.

**Expected effect.** The context head can be scored, but its mistakes never
alter training state. Autonomous context errors are therefore absent from the
training distribution.

**Executable regression.**
`test_model_context_action_changes_the_shared_transition`.

**Correction.** Context actions have their own teacher-forcing fraction and
source record. Model-sourced reads use model logits and the exact remaining
context budget.

### A03 — mixed evidence updates are oracle actions

**Confirmed.** After frontier selection, mixed rollout intersects the selected
indices with `supervision.include_as_evidence` (`training.py:857-868`).

**Expected effect.** Model evidence false positives and false negatives do not
affect the neural evidence state during training.

**Executable regression.**
`test_model_evidence_action_changes_the_shared_transition`.

**Correction.** Evidence actions have an independent teacher-forcing fraction.
Model-sourced actions use calibrated evidence logits; oracle labels are used
only when that action source is explicitly oracle.

### A04 — termination observes different state

**Confirmed.** Mixed rollout computes termination from the current hypotheses
and evidence before applying context/evidence/frontier actions
(`training.py:769-831`); runtime calls the terminator after `step` has updated
hypotheses and evidence (`controller.py:543-580`). Oracle rollout is
post-transition but substitutes the old hypotheses when the next frontier is
empty (`training.py:646-650`).

**Expected effect.** The termination head learns the wrong temporal boundary
and cannot associate newly accumulated evidence with stopping.

**Executable regression.**
`test_termination_observes_post_transition_state_in_all_modes`.

**Correction.** The terminator always receives `ControllerTransition` output
after context refinement, evidence update, and next-frontier construction.
Empty frontiers remain canonical model inputs.

### A05 — termination controls are normalised differently

**Confirmed.** Oracle training divides the round by oracle trace length and
uses static zero-budget tests (`training.py:651-660`). Mixed rollout divides by
a separately derived `round_limit` and omits used-budget and frontier-empty
features (`training.py:788-794`). Runtime divides by configured maximum rounds
and actual resolved budgets (`controller.py:255-278`).

**Expected effect.** Identical controller states acquire different visible
representations, and oracle trace length leaks into a model feature.

**Executable regression.**
`test_termination_controls_are_byte_equal_across_execution_modes`.

**Correction.** A pure `termination_control_features` function uses only
configured limits and post-transition state.

### A06 — mixed controller limits are hard-coded

**Confirmed.** Model frontier selection uses `frontier_width=16` and
`hypotheses_per_node=2` (`training.py:843-849`) rather than
`SparseControllerConfig`.

**Expected effect.** Training does not reproduce deployed pruning when the
configured limits differ.

**Executable regression.**
`test_scheduled_actions_respect_configured_frontier_limits`.

**Correction.** All action policies receive the resolved controller
configuration from the shared controller instance.

### A07 — off-oracle frontier implies incomplete

**Confirmed.** Mixed rollout compares the set of current local nodes with an
entire recorded oracle frontier. A missing exact set match becomes
`UNKNOWN_INCOMPLETE` (`training.py:769-786`).

**Expected effect.** Recoverable deviations, partial valid frontiers,
duplicates, and converging hypotheses receive false stop supervision.

**Executable regressions.**
`test_state_oracle_continues_recoverable_off_oracle_frontier`,
`test_state_oracle_handles_partial_and_duplicate_frontiers`, and
`test_state_oracle_marks_unrecoverable_budget_state_incomplete`.

**Correction.** `StateOracle` labels the actual hypotheses, ledgers, budgets,
and expanded arcs. It asks whether a valid completion remains, rather than
matching one canonical wavefront.

### A08 — sparse evidence uses ordinary BCE

**Confirmed.** Evidence uses unweighted mean binary cross-entropy
(`losses.py:165-168`) and ordinary candidate accuracy is logged prominently.

**Expected effect.** Sparse negatives dominate optimisation and permit high
accuracy with negligible evidence recall.

**Executable regressions.**
`test_balanced_evidence_loss_penalises_missed_positive` and
`test_evidence_metrics_report_average_precision_and_label_counts`.

**Correction.** v0.1 adds configurable class-balanced/focal BCE, a set-level
positive-mass/soft-recall term, label counts, average precision, and
development-only threshold calibration.

### A09 — evidence is coupled to frontier retention

**Confirmed.** Runtime computes evidence only from `selected`, the next
frontier (`controller.py:439-476`). Mixed rollout has the same coupling
(`training.py:857-868`).

**Expected effect.** Terminal evidence must be needlessly expanded and can be
lost when frontier top-k pruning removes it.

**Executable regression.**
`test_evidence_can_be_included_without_frontier_expansion`.

**Correction.** Evidence and frontier candidate indices are independent in
`ControllerActions`; evidence updates precede next-hypothesis construction.

### A10 — top-k cannot deliberately choose no expansion

**Confirmed.** Every depth-eligible candidate enters stable top-k regardless
of expand-logit sign (`controller.py:369-397`). The expand logit merely adds a
negative `logsigmoid` term to priority.

**Expected effect.** The controller traverses even when the model rejects
every action, causing invalid expansions and hiding deliberate exhaustion.

**Executable regression.**
`test_model_policy_can_choose_an_empty_frontier`.

**Correction.** Model actions filter by a configurable expansion-probability
threshold before deterministic top-k. A learned null action remains a deferred
ablation.

## Root cause

Spider v0 had three independently written state machines: oracle rollout,
mixed rollout, and runtime. The individual differences above are symptoms of
that duplication. Spider v0.1 fixes the root cause by making proposal,
action selection, transition application, controls, budgets, ledgers, and the
termination observation point shared code.
