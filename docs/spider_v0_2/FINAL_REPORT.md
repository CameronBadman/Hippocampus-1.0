# Spider v0.2 Final Report

## Result

Spider v0.2 answers its narrow, post-sealed question:

> Does the tied recurrent multi-set Spider earn its additional complexity over
> symmetric pooled processing when learned stopping cannot terminate traversal
> early?

No. The recurrent model lost to the pooled control on all three matched seeds.
Its mean fixed-horizon structural success was `0.3854`, versus `0.4531` for
pooling. The mean paired recurrent-minus-pooled delta was `-0.0677`; the frozen
success rule required at least `+0.0200` and recurrent wins on at least two
seeds.

The recurrent path state is nevertheless causally used. Resetting it reduced
structural success to zero on every seed, and shuffling states between
hypotheses caused a mean degradation of `0.2682`. Thus the negative
architecture result is not explained by a completely ignored recurrent state.

This was a development-only architectural diagnostic. It did not open a new
sealed split or revisit the historical Spider v0/v0.1 sealed result.

## Registered comparison

All six accepted runs used:

- dataset `spider-programs-v0.3-recurrence-dev`;
- all 512 training and 128 validation cases;
- 6,000 optimizer steps and 24,000 examples per run;
- FP32 on A100;
- identical controller, renderer, action schedule, optimizer, and loss
  configuration;
- oracle-required horizons of four to eight rounds;
- learned frontier and evidence actions;
- no intermediate learned stopping;
- the frozen model source
  `acb533666d481daf9b6fb56562d69a5dd78c5e0e`.

The pooled control still carries a recurrent path representation. It replaces
multi-set attention with symmetric mean/max pooling and an MLP transition.
Therefore this experiment rejects an advantage for the recurrent **multi-set
processor**, not recurrence or vector manifolds in general.

## Structural and evidence results

| Seed | Recurrent | Pooled | Recurrent − pooled |
| ---: | ---: | ---: | ---: |
| 1701 | 0.3828 | 0.4609 | -0.0781 |
| 1802 | 0.3750 | 0.4453 | -0.0703 |
| 1903 | 0.3984 | 0.4531 | -0.0547 |
| Mean | 0.3854 | 0.4531 | -0.0677 |

| Aggregate metric | Recurrent | Pooled |
| --- | ---: | ---: |
| Exact evidence set | 0.3854 | 0.4531 |
| Evidence recall | 0.6198 | 0.7161 |
| Exact valid path rate | 1.0000 | 1.0000 |
| Mean arcs scored | 17.9063 | 17.9063 |
| Mean rounds | 5.9688 | 5.9688 |
| Mean per-case evaluation latency | 0.1552 s | 0.1018 s |
| Parameters | 297,049 | 200,549 |

Pooling is both more accurate on the primary metric and smaller/faster in this
diagnostic. The benchmark requires remembering one opaque binding and comparing
it later; a compact pooled recurrent state appears sufficient. The stronger
claim that attention is intrinsically unnecessary is not supported.

## Direct state-use interventions

| Intervention | Mean structural result | Mean degradation | Seeds degraded by at least 0.05 |
| --- | ---: | ---: | ---: |
| Intact | 0.3854 | 0.0000 | 0/3 |
| Detach between rounds | 0.3854 | 0.0000 | 0/3 |
| Reset every round | 0.0000 | 0.3854 | 3/3 |
| Shuffle between hypotheses | 0.1172 | 0.2682 | 3/3 |
| Replace with pooled current node | 0.0755 | 0.3099 | 3/3 |

Detach is expected to preserve checkpoint inference because it changes gradient
flow, not forward values. Reset and graph-local shuffle are the registered
causal forward ablations. Both pass the frozen material-state-use rule.

The correct conclusion is:

1. long-horizon information is present in the recurrent state;
2. termination previously prevented that information from affecting the
   ordinary autonomous metric;
3. once stopping is suppressed, the multi-set recurrent processor still does
   not outperform a simpler pooled recurrent transition.

## Termination

Fixed-horizon execution removed the one-round stopping collapse from the
architectural comparison: every run completed its required horizon, valid path
and trace rates were `1.0`, and semantic invalid expansion was zero.

It did not repair learned termination. Mean final autonomous success was
`0.0026` for recurrent and `0.0000` for pooled; five runs scored zero and the
remaining run answered one of 128 cases. The models overwhelmingly mapped the
final state to `UNKNOWN_INCOMPLETE`.

The implemented factorised termination and learned null-expansion mechanisms
remain excluded from this frozen comparison. They are appropriate future
development hypotheses:

- predict evidence sufficiency separately from useful work remaining;
- permit branch-level null expansion without executing global termination;
- stop only for sufficient evidence, exhausted useful work, or exact budget
  exhaustion.

They require their own non-sealed experiment and must not be used to rewrite
this result.

## Integrity and reproducibility

- Six accepted runs and 36,000 total optimizer steps were aggregated.
- All runs used the frozen train and validation hashes.
- The fail-closed aggregator verified all 42 checkpoint/archive Drive records.
- Every run reports zero sealed access.
- Deterministic replay mismatches: zero.
- Row-permutation decision mismatches: zero.
- Checkpoint ZIPs, manifests, and standalone checkpoints passed SHA-256 and
  payload verification.
- The complete host CPU/CUDA suite passes 177 tests.
- Every A100 run independently recorded 158 passing remote tests before
  training.
- Google Drive backup:
  [Spider-v0.2-Fixed-Horizon](https://drive.google.com/drive/folders/1A8QnvZKDSWeiTXvi6RwYx76LFVAcDZRw).

Machine-readable evidence is in:

- `artifacts/spider_v0_2/training/TRAINING_SUMMARY.json`;
- `artifacts/spider_v0_2/training/training_experiments.jsonl`;
- `artifacts/spider_v0_2/GOOGLE_DRIVE_BACKUP.json`;
- `artifacts/spider_v0_2/plots/`.

## Research decision

Do not increase model size or proceed to compositional edge attention on the
strength of this result. The multi-set recurrent processor has not earned its
cost on the registered recurrence-necessity task.

Retain the pooled recurrent model as the control. The next bounded work should
separate two questions:

1. can the factorised terminator and null-expansion action restore autonomous
   final decisions without suppressing useful traversal;
2. is there a matched development task that genuinely requires maintaining
   several distinct exchangeable observations, rather than one compact binding
   that symmetric pooling can preserve?

Neither question should reopen a sealed set. This synthetic result validates
deterministic long-horizon state use but is not evidence of production,
natural-language, or broad graph-reasoning capability.
