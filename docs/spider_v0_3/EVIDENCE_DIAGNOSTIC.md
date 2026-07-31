# Spider v0.3 preserved-checkpoint evidence diagnostic

## Result

The exact evidence funnel was run on all six immutable Spider v0.2
fixed-horizon checkpoints and all 128 recurrence-necessity validation cases.
This is a post-v0.2 diagnostic and does not alter the historical result,
checkpoint, operating threshold, or sealed policy.

Every required evidence action was reachable, enumerated, and scored in every
run. The measured failure therefore occurs after candidate coverage.

| Model | Seed | Reachable | Scored | Selected/scored | AP | Exact set | FP/case |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled | 1701 | 1.0000 | 1.0000 | 0.6250 | 0.6823 | 0.4141 | 0.4062 |
| pooled | 1802 | 1.0000 | 1.0000 | 0.7188 | 0.5553 | 0.3984 | 0.6875 |
| pooled | 1903 | 1.0000 | 1.0000 | 0.7656 | 0.4948 | 0.3828 | 0.7422 |
| recurrent | 1701 | 1.0000 | 1.0000 | 0.5938 | 0.4772 | 0.3281 | 0.5625 |
| recurrent | 1802 | 1.0000 | 1.0000 | 0.6016 | 0.4961 | 0.3125 | 0.5547 |
| recurrent | 1903 | 1.0000 | 1.0000 | 0.5859 | 0.5769 | 0.3750 | 0.3906 |

Across pooled seeds, conditional selection recall averaged 0.7031, evidence
average precision 0.5775, exact-set accuracy 0.3984, and false positives 0.6120
per case. The registered diagnostic targets of 0.95 scored-positive coverage
and 0.90 conditional selection recall therefore produce a split decision:
coverage passes; selection does not.

## Interpretation

The existing fixed-horizon controller does not lose required evidence because
frontier expansion fails to enumerate it or because the search budget removes
it before scoring. Required candidates are instead insufficiently separated
from converging, structurally plausible distractor edges, and the preserved
threshold admits substantial false-positive evidence.

This supports proceeding with the bounded E0/E1/E2 evidence-objective
comparison and disjoint exact-set calibration. It does not establish that the
registered ranking objective will improve the result.

The controller invariant that evidence may be recorded without frontier
retention is covered by an executable regression test. Evidence and traversal
selection are separate actions in the measured pipeline.

## Reproducibility

The complete edge-specific observations grouped by case, family, horizon, and
round are in
`artifacts/spider_v0_3/preserved_diagnostics/diagnostics.jsonl`. The concise
generated table is in
`artifacts/spider_v0_3/preserved_diagnostics/SUMMARY.md`.

Run:

```bash
.venv/bin/python scripts/diagnose_spider_v0_2_evidence.py
```

The script validates each checkpoint against its historical SHA-256 digest
before evaluation and writes only to the Spider v0.3 artifact tree.
