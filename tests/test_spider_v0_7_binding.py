from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys

import torch

from hippocampus import GraphSchema

from hippocampus.programs import (
    V07_DATASET_VERSION,
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    audit_aligned_program_labels,
    default_binding_specs,
    generate_binding_cases,
    observable_symbols,
    pack_rendered_cases,
    verify_case,
)
from hippocampus.spider import (
    CanonicalBindingEvidenceReadout,
    PooledScorer,
    SpiderModelConfig,
    load_experiment,
)
from hippocampus.spider.types import CandidateReadoutContext, PaddedSet


ROOT = Path(__file__).resolve().parents[1]


def _phase_b_module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = ROOT / "scripts/spider_v0_4_phase_b.py"
    spec = importlib.util.spec_from_file_location("spider_v07_phase_b", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generator() -> GraphProgramGenerator:
    return GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=8,
            max_nodes=8,
            min_path_length=1,
            max_path_length=1,
            generator_version=V07_DATASET_VERSION,
            lookup_binding_mode="matched_conjunction",
        )
    )


def _outgoing(case):
    start = case.start_nodes[0]
    return tuple(edge for edge in case.edges if edge.source_node == start)


def _scalar_inventory(case) -> Counter[float]:
    return Counter(
        atom.scalar
        for edge in case.edges
        for atom in edge.atoms
        if atom.scalar is not None
    )


def test_matched_lookup_requires_relation_value_and_validity() -> None:
    case = _generator().generate(
        family=ProgramFamily.LOOKUP,
        seed=71,
        answerable=True,
    )
    query_symbols = {
        symbol for atom in case.query_atoms for symbol in atom.symbols
    }
    candidates = _outgoing(case)

    assert len(candidates) == 4
    assert {len(edge.atoms) for edge in candidates} == {2}
    assert (
        len(
            {
                len(case.nodes[edge.destination_node].summary_atoms)
                for edge in candidates
            }
        )
        == 1
    )

    patterns: Counter[tuple[bool, bool, bool]] = Counter()
    for edge in candidates:
        edge_symbols = {
            symbol for atom in edge.atoms for symbol in atom.symbols
        }
        destination_symbols = {
            symbol
            for atom in case.nodes[edge.destination_node].summary_atoms
            for symbol in atom.symbols
        }
        patterns[
            (
                bool(query_symbols & edge_symbols),
                bool(query_symbols & destination_symbols),
                edge.valid,
            )
        ] += 1

    assert patterns == Counter(
        {
            (True, True, True): 1,
            (True, False, True): 1,
            (False, True, True): 1,
            (True, True, False): 1,
        }
    )
    assert sum(
        target.include_as_evidence
        for target in case.trace.rounds[0].candidates
    ) == 1
    assert verify_case(case).valid


def test_lookup_negative_is_metadata_matched_and_breaks_the_conjunction() -> None:
    generator = _generator()
    positive = generator.generate(
        family=ProgramFamily.LOOKUP,
        seed=73,
        answerable=True,
    )
    negative = generator.generate(
        family=ProgramFamily.LOOKUP,
        seed=73,
        answerable=False,
    )

    assert positive.query_atoms == negative.query_atoms
    assert observable_symbols((positive,)) == observable_symbols((negative,))
    assert _scalar_inventory(positive) == _scalar_inventory(negative)
    assert [len(node.summary_atoms) for node in positive.nodes] == [
        len(node.summary_atoms) for node in negative.nodes
    ]
    assert [
        (edge.source_node, edge.destination_node, len(edge.atoms))
        for edge in positive.edges
    ] == [
        (edge.source_node, edge.destination_node, len(edge.atoms))
        for edge in negative.edges
    ]
    assert all(
        not target.include_as_evidence
        for target in negative.trace.rounds[0].candidates
    )
    assert verify_case(positive).valid
    assert verify_case(negative).valid


def test_v0_7_partitions_are_versioned_and_symbol_disjoint() -> None:
    specs = default_binding_specs()

    assert [spec.case_count for spec in specs] == [8192, 512, 512, 1024]
    assert all(spec.dataset_version == V07_DATASET_VERSION for spec in specs)
    assert all(not spec.sealed for spec in specs)

    seen: set[str] = set()
    for spec in specs:
        cases = generate_binding_cases(spec, limit=128)
        symbols = observable_symbols(cases)
        assert symbols
        assert seen.isdisjoint(symbols)
        seen.update(symbols)

        lookup = [case for case in cases if case.family is ProgramFamily.LOOKUP]
        assert lookup
        audit = audit_aligned_program_labels(cases)
        assert audit.invalid_case_count == 0
        assert audit.evidence_label_mismatch_count == 0


def _flatten(values):
    return tuple(item for owner in values for item in owner)


def test_binding_targets_follow_seeded_row_permutations() -> None:
    case = _generator().generate(
        family=ProgramFamily.LOOKUP,
        seed=79,
        answerable=True,
    )
    schema = GraphSchema(summary_dim=16, context_dim=16, edge_dim=16)
    renderer = SyntheticManifoldRenderer(
        schema,
        query_dim=16,
        seed=91337,
        geometry="orthogonal_aligned",
    )
    pair_counts: list[int] = []
    pair_tensors = []
    for row_seed in (11, 29):
        rendered = renderer.render(case, row_permutation_seed=row_seed)
        packed = pack_rendered_cases(
            (case,),
            (rendered,),
            schema=schema,
        )
        targets = packed.binding_targets
        assert targets is not None
        summary_symbols = _flatten(rendered.summary_row_symbols)
        edge_symbols = _flatten(rendered.edge_row_symbols)
        for query_row, summary_row in targets.query_summary_pairs.tolist():
            assert set(rendered.query_row_symbols[query_row]).intersection(
                summary_symbols[summary_row]
            )
        for query_row, edge_row in targets.query_edge_pairs.tolist():
            assert set(rendered.query_row_symbols[query_row]).intersection(
                edge_symbols[edge_row]
            )
        pair_counts.append(targets.pair_count)
        pair_tensors.append(
            (
                targets.query_summary_pairs.clone(),
                targets.query_edge_pairs.clone(),
            )
        )

    assert pair_counts[0] == pair_counts[1]
    assert any(
        not first.equal(second)
        for first, second in zip(pair_tensors[0], pair_tensors[1], strict=True)
    )


def _padded(values: torch.Tensor) -> PaddedSet:
    return PaddedSet(
        values,
        torch.ones(values.shape[:2], dtype=torch.bool),
    )


def test_canonical_binding_readout_is_row_permutation_invariant() -> None:
    torch.manual_seed(83)
    readout = CanonicalBindingEvidenceReadout(
        d_model=8,
        control_width=6,
    )
    path = torch.randn(3, 4, 8)
    context = CandidateReadoutContext(
        query=_padded(torch.randn(3, 3, 8)),
        source=_padded(torch.randn(3, 2, 8)),
        edge=_padded(torch.randn(3, 3, 8)),
        destination=_padded(torch.randn(3, 4, 8)),
        global_evidence=_padded(torch.randn(3, 2, 8)),
        controller_features=torch.randn(3, 6),
    )

    expected = readout(path, context)
    permuted = CandidateReadoutContext(
        query=_padded(context.query.values[:, [2, 0, 1]]),
        source=context.source,
        edge=_padded(context.edge.values[:, [1, 2, 0]]),
        destination=_padded(context.destination.values[:, [3, 1, 0, 2]]),
        global_evidence=context.global_evidence,
        controller_features=context.controller_features,
    )

    assert torch.allclose(expected, readout(path, permuted), atol=1e-6)


def test_binding_alignment_loss_is_finite_and_differentiable() -> None:
    case = _generator().generate(
        family=ProgramFamily.LOOKUP,
        seed=89,
        answerable=True,
    )
    schema = GraphSchema(summary_dim=16, context_dim=16, edge_dim=16)
    renderer = SyntheticManifoldRenderer(
        schema,
        query_dim=16,
        seed=91337,
        geometry="orthogonal_aligned",
    )
    batch = pack_rendered_cases(
        (case,),
        (renderer.render(case, row_permutation_seed=17),),
        schema=schema,
    )
    model = PooledScorer(
        SpiderModelConfig(
            summary_dim=16,
            context_dim=16,
            edge_dim=16,
            query_dim=16,
            d_model=32,
            num_heads=4,
            num_blocks=1,
            path_rows=4,
            evidence_rows=4,
            evidence_readout="canonical_binding",
        )
    )

    loss, pair_count = model.binding_alignment_loss(batch, temperature=0.07)

    assert pair_count == batch.binding_targets.pair_count
    assert pair_count > 0
    assert torch.isfinite(loss)
    loss.backward()
    readout = model.evidence_readout
    assert isinstance(readout, CanonicalBindingEvidenceReadout)
    assert all(
        parameter.grad is not None
        for parameter in readout.canonicalizer_parameters()
    )


def test_v0_7_configs_isolate_the_registered_alignment_variable() -> None:
    experiments = {
        arm: load_experiment(ROOT / f"configs/spider_v0_7/{arm}.json")
        for arm in ("R0", "R1", "R2")
    }

    assert all(
        experiment.raw["dataset"]["version"] == V07_DATASET_VERSION
        for experiment in experiments.values()
    )
    assert all(
        experiment.raw["dataset"]["fit_operating_policy"] is False
        for experiment in experiments.values()
    )
    assert experiments["R0"].model_config.evidence_readout == "dedicated_pooled"
    assert experiments["R1"].model_config.evidence_readout == "canonical_binding"
    assert experiments["R2"].model_config.evidence_readout == "canonical_binding"
    assert experiments["R0"].loss_config.binding_alignment == 0.0
    assert experiments["R1"].loss_config.binding_alignment == 0.0
    assert experiments["R2"].loss_config.binding_alignment == 0.1
    assert all(
        experiment.controller_config.evidence_selection_policy
        == "candidate_null"
        for experiment in experiments.values()
    )


def test_evidence_only_scope_freezes_the_transition_backbone() -> None:
    experiment = load_experiment(ROOT / "configs/spider_v0_7/R2.json")
    model = PooledScorer(experiment.model_config)
    module = _phase_b_module()

    names = module._configure_trainable_scope(model, "evidence")

    assert names
    assert all(
        name.startswith(
            ("evidence_readout.", "candidate_evidence_null_decoder.")
        )
        for name in names
    )
    assert not model.query_projection.weight.requires_grad
    assert not model.transition[1].weight.requires_grad
