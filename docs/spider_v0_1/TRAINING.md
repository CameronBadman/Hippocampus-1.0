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

## Realised local matrix

The ten accepted records ran on an NVIDIA GeForce RTX 5070 Ti in FP32 for
400 optimiser steps per trained record. All runs used all 512 training cases
and all non-sealed validation cases. The matrix source was commit `496b750`;
the sealed evaluator and frozen finalist were committed separately before
access.

The recurrent three-seed autonomous-success mean was `0.3224 ± 0.0402`
(population standard deviation). The matched pooled mean was
`0.2831 ± 0.0067`. The selected recurrent seed was 603, with validation
aggregate `0.3537` and evidence threshold `0.486353`, calibrated once on
validation ID.

The complete records are:

- `artifacts/spider_v0_1/experiments.jsonl`;
- `artifacts/spider_v0_1/EXPERIMENT_SUMMARY.md`;
- `artifacts/spider_v0_1/autoresearch/`; and
- `artifacts/spider_v0_1/FINALIST_MANIFEST.json`.

## Post-sealed Colab replication

The 400-step matrix is a controlled comparison, not a convergence study. A
separate Google Colab replication was therefore pre-registered after the
sealed result had been opened. The initial 2,000-step T4 attempt was
interrupted at the user's request while the remote workflow was being
investigated. It produced zero accepted records and is retained only as the
machine-readable abort record
`artifacts/spider_v0_1/COLAB_2K_ABORT.json`.

The replacement protocol trains recurrent and pooled models for 5,000 steps
on the same 512 non-sealed training cases, using three new seeds per model:
30,000 optimiser steps in total. H100 was preferred but unavailable to the
account; the registered A100 fallback was allocated and passed a real CUDA
preflight on an NVIDIA A100-SXM4-40GB. The preflight records driver 580.82.07,
CUDA 12.8, PyTorch 2.11.0+cu128, compute capability 8.0, and BF16 support. The
frozen runs remain FP32 for direct comparability.

These replications cannot alter the selected model, evidence threshold, or
sealed interpretation. Their exact protocol is
`artifacts/spider_v0_1/COLAB_5K_PROTOCOL.json`; allocation evidence is
`artifacts/spider_v0_1/COLAB_5K_ALLOCATION.json`. No sealed access is allowed
from the remote job.

The completed accepted matrix contains six isolated A100 runs and 30,000
optimizer steps. The recurrent scores were `0.3622`, `0.3778`, and `0.3778`
(`0.3726 ± 0.0074` population SD). The pooled scores were `0.4034`, `0.3821`,
and `0.3750` (`0.3868 ± 0.0121`). The post-sealed primary metric therefore
favored pooled by `0.0142`. This does not change the frozen finalist.

Every accepted run followed the same durability gate:

1. complete training and all non-sealed evaluation;
2. download the declared archive while the Colab lease remains active;
3. verify the archive SHA-256 and all 11 manifested files;
4. upload both the complete archive and standalone `checkpoint.pt` to Drive;
5. read back Drive IDs, parent folder, and byte counts;
6. record hashes and links in
   `artifacts/spider_v0_1/GOOGLE_DRIVE_BACKUP.json`; and
7. release the Colab session only after those checks pass.

The resulting Drive folder contains all six checkpoints, all six archives, the
JSONL experiment ledger, and JSON/Markdown summaries:
<https://drive.google.com/drive/folders/10Pmjb0lBATNtGWyf823SB4qHAYaZ7Euw>.
Regenerate the checked aggregate deterministically with:

```bash
.venv/bin/python scripts/aggregate_spider_v0_1_colab_runs.py
```

The original one-session job loss and the excluded concurrent-session retry
are preserved as infrastructure failures. Neither contributed an accepted
score. The final aggregate is
`artifacts/spider_v0_1/colab_5k/COLAB_5K_SUMMARY.json`.
