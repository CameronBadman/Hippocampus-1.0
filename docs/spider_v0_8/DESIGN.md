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

The candidate scorer is a position-free set Transformer over independently
fused candidate states. Its direct canonical similarity residual and learned
NULL energy make set selection calibration-free. Training uses graph-balanced
BCE, multi-positive listwise mass, and hard-negative ranking. T2 alone adds an
explicit multi-positive canonical-alignment loss; T1 has the same architecture
without that extra objective.

Candidate IDs and labels are never inference inputs. Candidate IDs are used
only as stable audit tie-breakers after scoring. Runtime validation rejects
supervisor fields, and model execution receives only query/incoming text
embeddings, memory text embeddings, runtime-visible attributes, relationship
edges, and controller-independent packed mappings.

The frozen comparison is intentionally small:

- T0: frozen MiniLM similarity and registered active/supersession controls;
- T1: packed canonical scorer without explicit alignment loss;
- T2: T1 plus the explicit multi-positive alignment loss.

The three learned arms share all parameters, data, optimization, and checkpoint
selection rules. No post-screen architecture edits were allowed.
