# Spider v0.1 AutoResearch result

## Completed protocol

All ten pre-registered records completed under source commit `496b750` and
dataset digest `101af9fd…a3105`. Every record used complete non-sealed
validation splits. There were no crashes, guard violations, deterministic
replay mismatches, row-permutation decision mismatches, or sealed accesses.

## Causal sequence

| Experiment | Autonomous success |
|---|---:|
| E000 old checkpoint, corrected controller | 0.0838 |
| E001 unified transition, oracle actions | 0.3636 |
| E002 scheduled closed loop | 0.2841 |
| E003 balanced/set evidence | 0.3011 |
| E004 hierarchical recurrent | 0.2656 |
| E005 matched hierarchical pooled | 0.2784 |

Alignment produced the largest improvement. Scheduled execution increased
mean rounds from 1.26 to 1.55 and reduced the one-round stop rate from 0.928
to 0.865, but worsened autonomous success, termination accuracy, and answered
risk. The evidence objective recovered some score and improved evidence F1
relative to E002. Hierarchical termination did not improve the primary seed.

## Three-seed comparison

| Model | Seed scores | Mean ± population SD |
|---|---|---:|
| recurrent hierarchical | 0.2656, 0.3480, 0.3537 | 0.3224 ± 0.0402 |
| pooled hierarchical | 0.2784, 0.2926, 0.2784 | 0.2831 ± 0.0067 |

The recurrent model wins the frozen primary metric by 0.0393 on the three-seed
mean, but is much more variable. The pooled model has better mean validation-ID
evidence F1 (0.5229 versus 0.4894), evidence recall (0.3980 versus 0.3810),
and answered risk (0.3643 versus 0.3935). Recurrence therefore improves the
strict aggregate success criterion without dominating every subsystem.

## Frozen finalist

The selected checkpoint is recurrent seed 603, the highest validation score
within the winning model family. Its evidence threshold is 0.486353, calibrated
once on validation ID. `FINALIST_MANIFEST.json` freezes its source, dataset
digest, config, checkpoint hash, threshold, and selection rule before v0.2
sealed access.

## Research conclusion before sealed evaluation

The train/runtime mismatch was causal, but not the only failure. Correct
alignment materially improves autonomous success while one-round stopping
remains common. The first scheduled curriculum and hierarchical head are not
individually supported by the primary-seed comparison. The remaining errors
appear to involve long-horizon termination/evidence credit and model variance,
not packed-graph execution.
