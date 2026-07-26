# Spider v0.2 recurrence-utility experiment

## Objective

Determine whether the existing recurrent path state improves long-horizon
structural graph execution once intermediate learned stopping is disabled.

## Immutable history

- `docs/spider_v0/`, `docs/spider_v0_1/`
- `artifacts/spider_v0/`, `artifacts/spider_v0_1/`
- all Spider v0 and v0.1 sealed results and selection decisions

No sealed split may be generated, loaded, calibrated on, or evaluated.

## Evaluator

`scripts/spider_v0_2_evaluator.py` must emit a finite JSON record and return
zero only when protocol, determinism, row-invariance, and sealed-access guards
pass. Scientific losses are accepted and retained.

## Metric

Primary: fixed-horizon structural success on non-sealed validation and the
recurrence-necessity validation split.

Secondary: final autonomous success, exact evidence set, evidence recall,
valid-path rate, semantic invalid expansion rate, rounds, arcs scored, and
contexts read.

## Search space

- model: preserved recurrent or pooled architecture;
- horizon: oracle-required, 4, 6, or 8;
- recurrent intervention: none, reset, detach, shuffle, or
  pooled-current-node;
- seed: 1701, 1802, 1903.

No model-width, renderer, attention, writer, or compositional-attention search
is allowed in the primary diagnostic.

## Decision rule

- Material state use: mean reset-or-shuffle degradation >= 0.05 and degradation
  in at least two of three seeds.
- Recurrent advantage: mean paired recurrent-minus-pooled structural success
  >= 0.02 and recurrent wins at least two of three seeds.

## Budget

- Existing-checkpoint diagnostic: all accepted runs; no training.
- Recurrence-necessity training: at most six accepted runs, three seeds per
  model, with matched case and step budgets.
- Crashes and guard failures remain in the ledger and do not count as accepted
  runs.

## Logging

Append every attempt to
`artifacts/spider_v0_2/experiments.jsonl`; generate
`artifacts/spider_v0_2/EXPERIMENTS.md`. Never delete negative results.

## Stop policy

pause_every: never

Stop after the fixed experiment matrix and factorised-termination/null-action
implementation are verified, or when a reproducible infrastructure failure
prevents further execution.

