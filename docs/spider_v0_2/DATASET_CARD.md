# Recurrence-Necessity Development Dataset

## Purpose

`spider-programs-v0.3-recurrence-dev` tests whether a path processor carries
and uses information across four to eight graph transitions. It is a
development-only diagnostic dataset. It has no sealed split and must not be
presented as an independent final evaluation.

## Construction

Each case has three equal-length branches. A branch observes an opaque binding
token on its first edge. Intermediate node summaries and edges are matched
across branches. Every final edge reaches the same answer node, but exactly one
final edge repeats the token observed at the beginning of its own path. That
history-consistent edge is the exact evidence action.

The paired case preserves:

- topology and latent node/edge identities;
- node observations;
- query observations;
- edge insertion order;
- the global multiset of edge observations;
- degree and manifold-cardinality profiles.

It permutes the first-edge token assignment so a different final arc becomes
the exact evidence action. Thus a local final-arc scorer observes the same
candidate set in both cases; only the carried path history distinguishes the
label.

All branches are acceptable frontier actions through the final round. Oracle
termination is `CONTINUE` before the required horizon and `ANSWER` afterward.
The exact supervisor records both the answer node and required evidence edge.
Evidence-edge IDs are latent labels and are never rendered.

## Splits

| Split | Cases | Seed namespace | Horizons |
|---|---:|---:|---:|
| `train_recurrence_necessity` | 512 | 810000 | 4–8 |
| `validation_recurrence_necessity` | 128 | 820000 | 4–8 |

Cases occur in matched pairs. Split case counts and any evaluation limits must
therefore be even.

## Leakage controls

- Correct branch is randomized by seed.
- Edge insertion order is randomized.
- All first-hop neighbours have identical summary values, row counts, and
  out-degree.
- All branch paths have identical length, degree, and edge-cardinality
  profiles.
- The correct final edge has no stable insertion position.
- The paired views share all local observation multisets.
- No node ID, edge ID, branch index, oracle cost, horizon, or answer label is
  rendered as a model feature.

The generated leakage report measures the best constant final-candidate
position heuristic. The frozen guard requires it to remain below 0.45 for
three branches (chance is one third), and requires zero first-hop profile
mismatches.

## Limitations

This split isolates one narrow kind of state use: remembering and comparing an
opaque observation across a path. It does not establish natural-language
reasoning, broad algorithmic generalisation, calibrated stopping, or
production utility. Because every case is answerable, it is not a standalone
abstention benchmark. Existing non-sealed v0.2 validation splits remain the
termination and unknown-reason diagnostics.

