# Spider v0.4 Final Report: Representation and Exact Evidence Recovery

## Conclusion

Spider v0.4 did not produce a deployable improvement over the pooled evidence
control. It did isolate the failure more sharply:

- cross-modal identity can be made perfectly identifiable without using the
  same coordinates in every modality;
- changing renderer geometry alone did not reliably improve graph-model
  evidence recovery;
- dedicated and row-aware evidence readouts did not expose a hidden Spider
  advantage;
- oracle cardinality reveals a large exact-set ceiling, but the learned null
  and cardinality decoders did not recover it.

The selected result remains F0, the pooled model with calibrated global
threshold. Its three-seed mean is 0.7549 exact evidence-set accuracy, 0.8830
precision, 0.7516 recall, and 0.9992 scored-positive coverage. Because mean
precision is below the required 0.90, this is a retained control—not a
successful final model.

## Evidence by phase

### A: representation is identifiable

The historical independent geometry A0 remained at chance on unseen-symbol
cross-modal retrieval. A1 shared-additive and A2 fixed orthogonal-aligned both
reached 1.000 AUROC, Top-1@64, and Top-1@256. A2 therefore passed its gate
without collapsing modalities to identical vectors.

### B: geometry alone is not causal

Under the fixed pooled model and 512-case protocol, A2/B2 won zero of three
matched seeds. Its mean lookup recall was effectively zero, and reachability
effects reversed sign across seeds. Phase C's larger/online-data comparison
was correctly skipped by its advancement gate.

### Audit and dataset amendment

Mechanical verification found no corrupt lookup or evidence labels across
10,240 cases. It did find that `UNKNOWN_UNSUPPORTED` was encoded by an extra
unique query atom, leaking answerability through query cardinality. Historical
v0.4 results were preserved. Evidence experiments moved to the versioned,
non-sealed `spider-programs-v0.4.1-aligned-evidence-dev` amendment, which
excludes that deferred termination family and restores the metadata baseline
to chance.

### D: evidence readout is not the main bottleneck

Neither a dedicated pooled evidence head nor a slot-aware Spider head passed
its matched-seed gate. D4's direct row access did not beat D3 or the shared
Spider control. The experiment therefore provides no evidence that useful
row-level path state is being discarded primarily by the current mean-pooled
action head.

At 2,000 steps, pooled D0 remained ahead of recurrent D2: 0.7549 versus 0.7305
exact set, and 0.7516 versus 0.7019 recall. This comparison does not show a
recurrent advantage.

### E/F: threshold ceiling exists, learned decoding fails

On frozen D0 candidates, oracle-cardinality top-k raises exact set from 0.7549
to 0.9274, a gain of 0.1725. Every matched seed exceeded the 0.15 diagnostic
gate, selecting the learned set-decoding branch.

The deployable experiments did not reproduce that result:

| Policy | Exact set | Precision | Recall | Cardinality MAE | Advances |
| --- | ---: | ---: | ---: | ---: | --- |
| global threshold (F0) | 0.7549 | 0.8830 | 0.7516 | 0.2682 | retained control |
| learned null (F1) | 0.7633 | 0.8688 | 0.7700 | 0.2881 | no, 0/3 |
| learned cardinality (F2) | 0.4805 | 0.5902 | 0.4932 | 0.4600 | no, 0/3 |
| null + cardinality (F3) | 0.5687 | 0.8773 | 0.4135 | 0.4456 | no, 0/3 |

No arm met the registered exact-set and recall gains on two seeds. A100
replication was conditional on an advancing learned decoder, so it was not run.

## Answers to the research questions

1. **Can the renderer preserve identity across modalities without making all
   coordinates identical?** Yes. A2 passes unseen-symbol retrieval perfectly
   through fixed modality-specific orthogonal transforms. This proves the
   interface is identifiable, not that graph reasoning is solved.
2. **Does corrected representation geometry improve lookup and reachability
   recall?** Not reliably under the tested pooled processor and data protocol.
   The renderer-only causal gate failed on all three seeds.
3. **Does Spider contain useful row-level evidence state discarded by mean
   pooling?** The registered D4 test provides no such evidence. Slot-aware
   readout did not outperform dedicated or shared pooled readout.
4. **Is exact-set failure caused by a global threshold once ranking is
   adequate?** Partly as an oracle diagnostic, but not yet as a learned
   solution. Oracle cardinality has a large ceiling; learned cardinality and
   null policies generalise poorly and damage precision or recall.

## What this does not establish

- It does not show that vector manifolds fail; every arm uses them.
- It does not show that recurrence can never help; this campaign deliberately
  holds the pooled processor for the set-decoder branch and earlier readout
  comparisons remain limited to the synthetic benchmark.
- It does not validate natural-language writers, production retrieval, or
  real-world memory behavior.
- It does not justify larger models, more Spider blocks, compositional edge
  transforms, learned termination, or a sealed evaluation.

## Recommended next experiment

Retain F0 and freeze the current v0.4 campaign. Before changing candidate
ranking or model capacity, instrument the learned cardinality head directly:
cardinality confusion by family, answerability, round, accumulated-evidence
count, and available candidates. Train that head on a balanced frozen-state
corpus while traversal and evidence logits remain frozen, and require strong
held-out cardinality accuracy before another controller evaluation.

If cardinality remains unlearnable from the current global state, the next
architectural question is the state representation supplied to set decoding,
not a larger Spider. If cardinality becomes reliable but exact set remains
poor, proceed to the already specified clean hard-negative ranking branch.

## Reproducibility and integrity

All nine new accepted runs used matched seeds 1701, 1802, and 1903, the same
dataset hash, fixed renderer, model dimensions, controller schedule, and
2,000-step limit. Every accepted run has zero deterministic replay mismatches,
zero row-permutation decision mismatches, finite metrics, and zero sealed
access. Raw checkpoints remain local/ignored; their hashes and selected steps
are recorded in the experiment ledger and finalist manifest.

Start with the machine summary at
`artifacts/spider_v0_4/phase_f/local_rtx5070ti/SUMMARY.json` and the generated
ledger beside it. Earlier phase reports remain immutable evidence and are not
reinterpreted by this conclusion.
