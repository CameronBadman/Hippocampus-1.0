# Spider v0.8 SRE transfer data card

The source is the read-only SRE Incident-Memory v3 corpus in the sibling
`hippo-qwen-2` checkout. All incidents and language are synthetic; Qwen3-14B
rendered text while deterministic state machines supplied labels.

| Partition | Cases | Use |
| --- | ---: | --- |
| Train | 1,600 | Parameter learning |
| Model selection | 100 | Checkpoint and arm selection |
| Development evaluation | 100 | One evaluation after selection |
| Public demo | 3 | Qualitative terminal demo only |

Each case has 64 candidate memories and 10–13 graph relationships. Required
evidence cardinality ranges from zero through four. Candidate pools include
query echoes, answer-shaped text, stale ownership, rolled-back deployments,
superseded fixes, contradictions, wrong-time and wrong-relationship memories,
near duplicates, relationship-path decoys, and mixed adversaries.

The upstream SRE sealed test is explicitly excluded. Dataset loaders and
campaign commands reject `test.inputs.jsonl` and `test.labels.jsonl`. No v0.8
claim may be interpreted as real-world SRE reliability or production
validation.
