# Spider v0.1 pre-registered training protocol

## Primary metric

The higher-is-better autonomous success rate is the fraction of cases meeting
all applicable conditions:

- answerable: `ANSWER`, valid traversal, and all required evidence recovered;
- unanswerable: exact unknown family and no false answer.

Teacher-forced ranking is diagnostic only. Selection ties are broken by:
evidence F1, lower false-answer rate, lower mean arcs scored, then experiment
ID.

## Fixed training setup

- Dataset: complete 512-case `spider-programs-v0.2` training split.
- Validation: complete non-sealed splits.
- Architecture: standard-attention tied recurrent Spider or pooled control,
  with matched controller/training machinery.
- Optimiser and parameter widths: frozen in `configs/spider_v0_1`.
- Maximum experiment time: five minutes per accepted exploration run.
- Guard: all CPU/CUDA tests applicable to the environment, finite metrics,
  zero replay mismatches, zero row-permutation decision mismatches, and no
  sealed access.
- Pause policy: never.

## Schedule

Five equal training phases use independent fractions:

| Phase | Frontier | Context | Evidence | Termination execution |
|---|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 0.75 | 0.75 | 0.75 | 1.00 |
| 3 | 0.50 | 0.50 | 0.50 | 0.75 |
| 4 | 0.25 | 0.25 | 0.25 | 0.50 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 |

The oracle-only control uses phase 1 for every step.

## Frozen matrix

| ID | Change |
|---|---|
| E000 | old selected v0 checkpoint, corrected controller, v0.2 validation only |
| E001 | recurrent standard, unified transition, oracle actions |
| E002 | E001 plus scheduled closed-loop actions |
| E003 | E002 plus balanced/set-level evidence objective |
| E004 | E003 plus hierarchical termination |
| E005 | pooled baseline with E004 controller, schedule, evidence, and termination |

After E000–E005, E004 and E005 receive two additional seeds each. This is a
fixed budget of ten accepted records. Crashes remain logged and do not consume
the accepted budget. No compositional attention, swapping, or other
architecture search is permitted.

## Calibration

Each trained run calibrates its evidence threshold on complete validation ID
only. The threshold then remains fixed for every OOD report. The pooled and
recurrent models use the same deterministic calibration rule.

## Sealed selection

The recurrent and pooled three-seed aggregates are computed first. The
finalist is selected by mean autonomous success, with the frozen tie-breakers.
Its exact source, dataset digest, threshold, seed, checkpoint hash, and
selection record are frozen before v0.2 sealed access. The sealed split is
opened once.
