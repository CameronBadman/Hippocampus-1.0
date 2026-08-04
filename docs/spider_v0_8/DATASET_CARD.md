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

The 1,600 training cases contain twelve incident scenario families. The
development partitions contain three different scenario families—clock skew,
regional failover, and secret rotation—so evaluation is scenario-family and
entity-lineage OOD under the same deterministic generator and text-rendering
pipeline. Related observations never cross partitions.

Each case has 64 candidate memories and 10–13 graph relationships. Required
evidence cardinality ranges from zero through four. Candidate pools include
query echoes, answer-shaped text, stale ownership, rolled-back deployments,
superseded fixes, contradictions, wrong-time and wrong-relationship memories,
near duplicates, relationship-path decoys, and mixed adversaries.

The upstream SRE sealed test is explicitly excluded. Dataset loaders and
campaign commands reject `test.inputs.jsonl` and `test.labels.jsonl`. No v0.8
claim may be interpreted as real-world SRE reliability or production
validation.

Frozen hashes and the exact model-selection/evaluation partition IDs are in
[`SOURCE_MANIFEST.json`](../../artifacts/spider_v0_8/SOURCE_MANIFEST.json) and
[`SPLIT_MANIFEST.json`](../../artifacts/spider_v0_8/SPLIT_MANIFEST.json).
