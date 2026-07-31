# Spider v0.3 Final Report

## Outcome

Spider v0.3 implemented the evidence funnel, disjoint calibration, exact CUDA
resume, directly labelled factorized termination, and per-hypothesis NULL
expansion. The registered local contingency matrix completed 12 evidence runs
and nine termination runs with zero sealed accesses and 211 passing CPU/CUDA
tests.

The scientific result is negative. Evidence ranking did not beat the existing
objective, longer evidence training did not improve recall, and neither
factorized termination nor branch-level NULL met the autonomous-controller
gate. T3 joint fine-tuning and the multi-binding architecture benchmark were
therefore not unlocked.

These runs used an NVIDIA GeForce RTX 5070 Ti because Colab rejected both A100
and H100 allocations despite valid OAuth credentials and no active official
sessions. They are complete hardware-qualified contingency evidence, not the
preregistered A100 result.

## Evidence pipeline

The preserved v0.2 checkpoint diagnostic found every required evidence item
reachable, enumerated, and scored. The pooled models averaged 0.703
selection recall conditional on scoring, 0.578 average precision, 0.398 exact
set accuracy, and 0.612 false positives per case. Candidate coverage was not
the bottleneck.

The registered 1,000-step screen produced:

| Arm | Recall | Precision | Exact set | Scored coverage | Decision |
|---|---:|---:|---:|---:|---|
| E0: current objective | 0.6250 | 0.9090 | 0.7031 | 0.9722 | Advance |
| E1: weighted BCE + ranking | 0.5139 | 0.9071 | 0.6510 | 0.9861 | Reject |
| E2: E1 + plausible negatives | 0.5139 | 0.9071 | 0.6510 | 0.9861 | Reject |

E1 lost 0.111 absolute recall and 0.052 exact-set accuracy on average without
a meaningful precision gain. It won zero matched seeds. E2 was byte-identical
to E1 because no extra plausible-negative target occurred in the collected
states; it is non-informative rather than a replication.

Only E0 advanced to 6,000 total steps. Its three-seed means were 0.576 recall,
0.828 precision, 0.641 exact-set accuracy, and 0.965 scored-positive coverage.
That is worse than its 1,000-step screen. The representative median seed was
1903, with:

- evidence recall 0.6042;
- precision 0.7838;
- exact-set accuracy 0.6250;
- scored-positive coverage 0.9792;
- conditional selection recall 0.6170; and
- calibrated raw evidence threshold 0.9995837.

Temperature scaling selected the maximum registered value, 4.0, in the
inspected full runs. The extreme raw thresholds and lower full-run recall show
overconfident evidence logits and training-duration degradation. The 0.90
conditional-recall diagnostic target was not met.

## Termination

Traversal, path-state updates, and evidence scoring were frozen. T0–T2 trained
fresh heads on detached oracle, mixed, model, premature-stop, and
budget-boundary controller states.

| Arm | Autonomous success | Fixed horizon | Retention | Continue recall | Premature stop | False answer |
|---|---:|---:|---:|---:|---:|---:|
| T0: hierarchical | 0.3490 | 0.6094 | 0.5726 | 0.6574 | 0.3426 | 0.1667 |
| T1: direct factors | 0.3490 | 0.6094 | 0.5726 | 0.6574 | 0.3426 | 0.1562 |
| T2: T1 + branch NULL | 0.3021 | 0.5052 | 0.5982 | 0.6574 | 0.3426 | 0.1042 |

T1 slightly reduced risk and false answers but did not change continuation or
mean autonomous success. T2 correctly allowed each hypothesis to die without
globally terminating, but it removed productive branches: false answers
declined while fixed-horizon and autonomous success also declined.

No seed met continuation recall 0.95, premature-stop rate below 0.25,
autonomous retention 0.85, and unknown macro recall 0.70. The selected
termination arm is therefore `null`.

## Answers to the milestone questions

**Was candidate coverage the main evidence problem?** No. Required items were
reachable and scored; ranking, calibration, and exact-set selection remained
the limiting stages.

**Did the bounded ranking objective improve recall or precision?** No. It
reduced recall and exact-set accuracy. The additional E2 mining path received
no training opportunities on this data.

**Did calibration repair evidence selection?** It produced a legal frozen
operating point, but exposed severe overconfidence and did not meet the recall
target.

**Did direct factor labels repair autonomous stopping?** No. Factor prediction
was learnable, but the executed decisions retained only 57% of fixed-horizon
success and stopped productive work too often.

**Did per-hypothesis NULL help?** It reduced false answers, but harmed
structural success and did not improve continuation recall.

**Should multi-set attention be tested now?** No. The preregistered controller
gate failed, so changing architecture would confound the unresolved control
problem.

## Verification and preservation

- Source tag: `spider-v0.3-evidence-source`
- Run source: `2d542860af20bfe6ef2ab64e2950df1d07ceb2da`
- Dataset hash:
  `0ed8e27ec44f3773f76b79f1947526f33ba233556b7db91fef04dcb647e5409d`
- Environment: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0
- Tests: 211 passed
- Sealed access count: zero in every run and aggregate
- Drive backup:
  [Spider v0.3 local contingency](https://drive.google.com/drive/folders/1fDeph0FQhW5fwV6V8Jn4CDJC6BmLFpWE)

All 21 final checkpoints are present in separate evidence and termination
Drive folders. The repository stores their source-qualified ledgers, detailed
metrics, byte sizes, local SHA-256 hashes, and Drive IDs. Spider v0.2 files and
its Drive folder were not modified.

## Limitations and next experiment

This does not show that vector manifolds fail; every compared model uses the
same manifold substrate. It does not establish a recurrent or attention
advantage. It also cannot be represented as an A100 replication.

Before another architecture comparison:

1. preregister an earlier evidence checkpoint or true validation-based stopping
   rule instead of assuming 6,000 steps is beneficial;
2. construct plausible-negative states mechanically and require a nonzero
   target count before admitting an E2 run;
3. diagnose why strong factor accuracies do not change executed continuation,
   including decision thresholds and rule precedence;
4. balance controller-state collection by useful-work and unknown-reason
   strata, then rerun T0/T1 only;
5. rerun the frozen matrix on one A100 when Colab grants the requested
   accelerator.

T3, multi-binding architecture experiments, and SRE/Entor transfer remain
deferred until pooled autonomous retention passes the registered gate.
