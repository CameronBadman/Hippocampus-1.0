# Spider v0.1 final report

## Outcome

Spider v0.1 removes the mechanically confirmed train/runtime controller
mismatches and substantially improves autonomous execution over the old
checkpoint diagnostic. In the frozen 400-step selection matrix, the tied
recurrent Spider beats the matched pooled control on the three-seed,
complete-validation primary metric.

That recurrent advantage did not survive the later, post-sealed 5,000-step
A100 replication. Across three new paired seeds, pooled scored `0.3868` versus
recurrent `0.3726`. One-round stopping also remained above `86%`. The combined
evidence supports train/runtime alignment as causal for the improvement over
v0, but does not support a durable recurrent advantage under longer training.

## Frozen experiment sequence

| Experiment | Autonomous success | Evidence F1 | Termination accuracy | Answered risk | One-round stop |
|---|---:|---:|---:|---:|---:|
| E000 old checkpoint, corrected runtime | 0.0838 | 0.3670 | 0.3281 | 0.4454 | 1.0000 |
| E001 unified transition, oracle actions | 0.3636 | 0.4196 | 0.3793 | 0.3819 | 0.9276 |
| E002 scheduled actions | 0.2841 | 0.4944 | 0.2898 | 0.3953 | 0.8651 |
| E003 balanced/set evidence | 0.3011 | 0.4990 | 0.3054 | 0.4358 | 0.9759 |
| E004 hierarchical recurrent | 0.2656 | 0.4974 | 0.2699 | 0.4086 | 0.8480 |
| E005 matched pooled | 0.2784 | 0.4897 | 0.2812 | 0.4093 | 0.8494 |

Subsystem metrics in this table are case-weighted across the seven complete
non-sealed validation splits. E000 is diagnostic only: its high raw evidence
recall came from indiscriminate inclusion and had poor exact-set accuracy.

## Frozen 400-step recurrent versus pooled

| Model | Three-seed scores | Mean ± population SD |
|---|---|---:|
| tied recurrent Spider | 0.2656, 0.3480, 0.3537 | 0.3224 ± 0.0402 |
| pooled control | 0.2784, 0.2926, 0.2784 | 0.2831 ± 0.0067 |

The recurrent mean is higher by `0.0393`. It is also higher on every reported
split mean:

| Split | Recurrent | Pooled |
|---|---:|---:|
| validation ID | 0.3099 | 0.2917 |
| graph-size OOD | 0.3021 | 0.2604 |
| path-length OOD | 0.3194 | 0.2674 |
| topology OOD | 0.3299 | 0.3021 |
| cardinality OOD | 0.3299 | 0.2743 |
| equivalent-view OOD | 0.3438 | 0.2778 |
| composition OOD | 0.3264 | 0.3056 |

This is not subsystem dominance. On validation ID, the pooled three-seed mean
had evidence F1 `0.5229` versus recurrent `0.4894`, evidence recall `0.3980`
versus `0.3810`, exact evidence-set accuracy `0.5417` versus `0.5130`, and
answered risk `0.3643` versus `0.3935`. The recurrent result is materially more
seed-sensitive.

## One-time sealed result

The finalist, recurrent seed 603, was frozen with checkpoint SHA-256
`a4d6fc…d71b87`, dataset digest `101af9…a3105`, and validation-ID evidence
threshold `0.486353`. The sole v0.2 sealed evaluation covered all 256 cases:

| Metric | Result |
|---|---:|
| primary autonomous success | 0.3672 |
| evidence precision / recall / F1 | 0.5063 / 0.4197 / 0.4589 |
| evidence average precision | 0.7593 |
| exact evidence-set accuracy | 0.5000 |
| termination accuracy | 0.4492 |
| unknown-reason accuracy | 0.4609 |
| exact valid-path rate | 0.2891 |
| answered coverage / risk | 0.3203 / 0.3171 |
| false-answer rate | 0.1016 |
| mean rounds / arcs scored / contexts read | 1.1367 / 3.3828 / 0.5313 |

Trace validity was `1.0000` and semantic invalid-expansion rate was `0.0000`.
Across 32 repeated/permuted samples there were zero deterministic replay
mismatches, zero decision mismatches, and maximum logit delta
`9.54e-7`.

Performance is uneven by family: corroboration autonomous success was
`0.6406`, reachability `0.3438`, lookup `0.3125`, and latest-valid `0.1719`.
Latest-valid evidence recall was zero. At final-decision level, CONFLICT and
UNSUPPORTED recall were zero; CONTINUE is not a terminal case label in that
matrix. Per-round behaviour still shows collapse: `93.36%` of cases stopped
after one round.

Teacher-forced sealed candidate Top-1 was `0.9595` and MRR `0.9797`, but those
figures are diagnostics only. They do not offset the autonomous failure modes.

## Answers to the milestone questions

### 1. Did alignment fix the one-round stopping collapse?

**Partly.** It fixed a large autonomous-performance collapse: E001 rose from
E000's `0.0838` to `0.3636` and reduced false-answer behaviour. It did not fix
the literal one-round collapse. E001 still stopped after one round on `92.76%`
of validation cases, and the selected model did so on `93.36%` of sealed
cases. Longer 5k training reduced the validation-ID rates to `86.98%`
recurrent and `86.46%` pooled, but that remains a severe collapse.

### 2. Did autonomous evidence recall improve?

**Evidence quality improved, but recall remains inadequate.** Relative to the
historical v0 sealed evidence F1 of `0.0232`, v0.2 sealed F1 is `0.4589`, with
recall `0.4197`, AP `0.7593`, and exact-set accuracy `0.5000`. Within the v0.1
causal sequence, balanced/set loss improved E002 evidence F1 from `0.4944` to
`0.4990` and recall from `0.3622` to `0.3716`. It did not exceed E000's raw
recall, which was inflated by false positives. The post-sealed 5k
validation-ID recall was `0.4116` recurrent and `0.4082` pooled, so longer
training did not produce a decisive evidence-recall advantage.

### 3. Did scheduled closed-loop training improve risk and termination?

**No for the registered schedule.** E002 reduced one-round stops and increased
mean rounds, but autonomous success fell from `0.3636` to `0.2841`,
termination accuracy from `0.3793` to `0.2898`, and answered risk worsened from
`0.3819` to `0.3953`. The schedule exposes the right states but needs a longer
or better-shaped curriculum.

### 4. Does recurrence now outperform the pooled baseline across seeds?

**Not under the longer post-sealed replication.** The frozen 400-step search
favored recurrent by `0.0393`, which determined the finalist before sealed
access. The independent 5,000-step A100 diagnostic instead favored pooled by
`0.0142`; pooled won two of three paired seeds and six of seven ID/OOD split
means. The original selection remains historically valid, but the stronger
follow-up falsifies the claim that recurrence reliably dominates this pooled
control. Only the frozen recurrent finalist was opened on sealed data; no
sealed pooled comparison was permitted.

### 5. Which failures now appear architectural rather than supervisory?

The exact state alignment defects are resolved and guarded by tests. Remaining
failures most suggest:

- a termination representation or decision problem: high candidate ranking
  coexists with almost universal early stop and collapsed unknown classes;
- insufficient long-horizon credit through discrete frontier/evidence choices;
- family-specific relational weakness, especially latest-valid context use;
- evidence selection losing positives between “scored” and “selected”; and
- high recurrent seed variance.

The negative scheduled and hierarchical ablations show that supervision is
not yet exhausted as an explanation. A longer curriculum and termination-loss
study should precede new edge-attention complexity.

## Post-sealed 5k A100 replication

The pre-registered follow-up used three new paired seeds, 512 non-sealed
training cases, 5,000 FP32 optimizer steps per run, and NVIDIA
A100-SXM4-40GB GPUs. It contributed 30,000 accepted optimizer steps. All six
accepted runs passed the remote test gate and evaluator guard, and every
reported sealed-access count was zero.

| Model | Seed scores | Mean ± population SD |
|---|---|---:|
| tied recurrent Spider | 0.3622, 0.3778, 0.3778 | 0.3726 ± 0.0074 |
| matched pooled control | 0.4034, 0.3821, 0.3750 | 0.3868 ± 0.0121 |

The paired recurrent-minus-pooled differences were `-0.0412`, `-0.0043`, and
`+0.0028`, for a mean difference of `-0.0142`.

| Split | Recurrent | Pooled | Difference |
|---|---:|---:|---:|
| validation ID | 0.3880 | 0.3672 | +0.0208 |
| graph-size OOD | 0.3021 | 0.3611 | -0.0590 |
| path-length OOD | 0.3507 | 0.3576 | -0.0069 |
| topology OOD | 0.3715 | 0.3715 | +0.0000 |
| cardinality OOD | 0.4097 | 0.4514 | -0.0417 |
| equivalent-view OOD | 0.3715 | 0.3889 | -0.0174 |
| composition OOD | 0.4097 | 0.4167 | -0.0069 |

On validation ID, recurrent retained lower answered risk (`0.2935` versus
`0.3514`) and higher termination accuracy (`0.4349` versus `0.3984`). Pooled
had slightly higher evidence F1 (`0.4903` versus `0.4870`) and exact
evidence-set accuracy (`0.5078` versus `0.4740`). Recurrent recall was only
slightly higher (`0.4116` versus `0.4082`). Neither model fixed early stopping:
the recurrent one-round stop rate was `0.8698` and pooled was `0.8646`.

The first multi-run Colab session was lost after one hour, so none of its
unrecovered outputs were accepted. A later attempt to run the final pair
concurrently lost the recurrent session before checkpoint creation; that
attempt is also excluded and recorded as infrastructure evidence. Every
accepted run used a fresh isolated A100 session and was downloaded and deeply
verified before release.

All six standalone checkpoints and complete run archives are stored in the
[verified Google Drive folder](https://drive.google.com/drive/folders/10Pmjb0lBATNtGWyf823SB4qHAYaZ7Euw).
The machine-readable results are
`artifacts/spider_v0_1/colab_5k/colab_5k_experiments.jsonl` and
`artifacts/spider_v0_1/colab_5k/COLAB_5K_SUMMARY.json`. This replication is
post-sealed evidence only and cannot alter the finalist, calibrated threshold,
or one-time sealed report.

## Implementation and verification

Oracle training, scheduled training, autonomous evaluation, and replay now use
one `propose -> label -> choose -> apply -> terminate` state machine.
Candidate and termination controls are pure shared functions. Context,
evidence, and frontier actions are independently scheduled. Evidence no
longer requires frontier retention, and a model frontier may be deliberately
empty.

Traversal uses `PackedTopology.expand_frontier`; summaries, edges, contexts,
and repeated owner occurrences use packed manifold gathers. No Python graph
engine was added to the hot path.

The final local gate passed 134 CPU tests and 6 CUDA tests on the RTX 5070 Ti.
The v0 report and artifacts were not modified or reopened. The v0.2 sealed
access marker and report are immutable artifacts.

## Limitations and next experiment

This is a small synthetic benchmark with a deterministic renderer. It does not
validate natural-language memory, learned writers, calibration, or production
reasoning. The 400-step selected runs establish a controlled comparison, not
full convergence.

The completed post-sealed A100 replication shows that longer training neither
removed one-round stopping nor reproduced the frozen recurrent advantage.
The next focused experiment should therefore compare a learned null-expansion
action and a factorised evidence-sufficiency terminator, with a training-length
matched recurrent/pooled control. It should not add compositional edge
attention until the termination failure is resolved.
