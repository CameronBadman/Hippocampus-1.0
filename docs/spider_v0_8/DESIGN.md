# Spider v0.8 design: SRE retrieval transfer

Spider v0.8 transfers one validated idea—an explicitly aligned canonical
matching space—onto synthetic natural-language incident memories. It does not
scale Spider or add answer generation.

Every SRE case is represented as a real `PackedGraph`: a query root, 64 memory
nodes, root-to-candidate retrieval arcs, and the runtime relationship graph.
The model enumerates candidates with `expand_frontier`, gathers their manifold
rows from packed storage, and forms graph-neighbour state through segmented
reductions. There is no parallel Python graph in scoring.

The query manifold contains frozen embeddings for the operator request and
incoming observation. Candidate summaries contain frozen memory-text rows.
Edge manifolds encode observable relationship type and effective time. A
small scorer projects query and memory rows into a canonical space, combines
row-level similarity, graph-neighbour state, and runtime-visible numeric
features, and emits a relevance energy plus a per-case NULL energy.

Training-only relevance alignment creates multi-positive contrastive targets.
Candidate IDs and labels are never inference inputs. The comparison isolates
this alignment loss against an otherwise identical packed scorer and a frozen
semantic baseline.
