# Spider v0.4 aligned development dataset

## Identity and purpose

`spider-programs-v0.4-aligned-dev` is a development-only synthetic graph
program dataset for isolating representation geometry, data reuse, evidence
readout, and evidence-set decoding. It does not replace or reinterpret any
v0-v0.3 dataset. It contains no sealed partition.

The four partitions are:

| Partition | Base cases | Use |
|---|---:|---|
| training | 8,192 | parameter learning |
| model selection | 512 | checkpoint selection only |
| calibration | 512 | temperature and evidence policy only |
| development evaluation | 1,024 | one-time accepted-arm evaluation |

The seed namespaces are disjoint. Case IDs, base-case IDs, and full generated
case content are hashed in `artifacts/spider_v0_4/splits/`. Related views must
be routed by `base_case_id`; the registered dataset currently materialises one
base view per group.

## Stratification

Every 128-case cycle crosses all combinations of:

- four program families;
- answerable versus unknown outcome;
- graph sizes 8, 16, 24, and 32;
- requested path lengths 1, 2, 3, and 4.

Required evidence cardinality and observed path length are also reported for
every partition. They cannot be made independent of program family and outcome
without changing task semantics: for example, a positive corroboration case
requires two sources while a positive lookup requires one, and lookup itself
is a one-hop program. The manifests therefore preserve identical joint
stratification across partitions and report the constrained observed
marginals, rather than claiming an impossible independent marginal balance.

## Presentation protocol

Training cases are not stored as fixed rendered tensors. A resumable lazy batch
source rerenders each selected case with a deterministic fresh row permutation
on every presentation. Presentation counters are part of the training
checkpoint, so an interrupted run resumes exactly. The 8,192-case arm provides
fresh case-local symbols; the later online arm will additionally generate new
base graphs continuously while holding total presented examples fixed.

Renderer A2 uses a shared latent symbol transformed by fixed seeded orthogonal
matrices per modality. No answer, path-membership, relation-ID, node-ID, row
position, or oracle control label is exposed to the model.

## Access policy

Model-selection data chooses checkpoints. Calibration data fits temperature
and the operating policy only after checkpoint selection. Development
evaluation is never used for either decision. Existing sealed splits are not
read, copied, or named as inputs by the v0.4 commands.

## Limitations

This benchmark is synthetic and does not establish natural-language or
production graph-retrieval validity. Orthogonal alignment deliberately makes
cross-modal identity learnable; the campaign tests whether that repair is
causal for evidence recovery, not whether a future writer will learn the same
geometry automatically.
