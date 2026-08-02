# Dataset amendment: v0.4.1 aligned evidence development

`spider-programs-v0.4.1-aligned-evidence-dev` is a correctness amendment for
subsequent Spider v0.4 evidence experiments. It does not alter
`spider-programs-v0.4-aligned-dev`, its hashes, or the completed Phase B result.

## Demonstrated issue

The v0.4 generator encoded `UNKNOWN_UNSUPPORTED` by appending one fresh opaque
query symbol. Each unsupported symbol appeared exactly once, so no reusable
unsupported concept existed. The appended row nevertheless made query
cardinality predict answerability with 0.666 accuracy versus a 0.500 majority
baseline.

That is both unhelpful supervision and a violation of the registered
cardinality-control rule.

## Narrow correction

Learned termination is deferred in this campaign. The evidence-only amendment
therefore removes unsupported-interface cases and alternates ordinary
absent/conflict cases with budget-incomplete cases. It changes neither the four
program-family semantics nor any answerable evidence label.

The amended corpus uses new seed namespaces and the same four partition sizes:
8,192 training, 512 model-selection, 512 calibration, and 1,024 development
evaluation. It retains exact family, outcome, graph-size, and requested-path
stratification, fresh row permutations, content hashing, and base-view grouping.

Generation is rejected unless every partition has:

- zero invalid cases;
- zero evidence-label mismatches;
- zero unsupported cases;
- query-cardinality answerability accuracy equal to its 0.500 majority
  baseline;
- disjoint case and base-case identities;
- zero sealed access.

All subsequent readout and decoding arms must name the v0.4.1 manifest hash.
Phase B remains historical evidence on v0.4 and is not retroactively compared
as though it used the corrected corpus.
