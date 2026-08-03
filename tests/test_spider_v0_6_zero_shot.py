from __future__ import annotations

from dataclasses import replace
import json
import random
from pathlib import Path

import pytest
import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    V06_DATASET_VERSION,
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    default_zero_shot_specs,
    generate_zero_shot_cases,
    observable_symbols,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ActionSchedule,
    CandidateEvidenceNullDecoder,
    ControllerState,
    PooledScorer,
    SparseControllerConfig,
    SparseWavefrontController,
    SpiderModelConfig,
    load_experiment,
)
from hippocampus.spider.types import CandidateOutputs
from hippocampus.spider.calibration import validate_calibration_source


ROOT = Path(__file__).resolve().parents[1]


def _candidate_outputs(count: int, *, d_model: int = 8) -> CandidateOutputs:
    path = torch.randn(count, 3, d_model, requires_grad=True)
    logits = [torch.randn(count, requires_grad=True) for _ in range(7)]
    return CandidateOutputs(path, *logits)


def _batch(*, row_seed: int = 0):
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    case = GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=10,
            max_nodes=10,
            min_path_length=2,
            max_path_length=2,
        )
    ).generate(
        family=ProgramFamily.REACHABILITY,
        seed=1_337,
        answerable=True,
        require_multiple_paths=True,
    )
    renderer = SyntheticManifoldRenderer(
        schema,
        query_dim=8,
        seed=811,
        geometry="orthogonal_aligned",
    )
    return pack_rendered_cases(
        (case,),
        (renderer.render(case, row_permutation_seed=row_seed),),
        schema=schema,
    )


def _model_config(*, candidate_null: bool = True) -> SpiderModelConfig:
    return SpiderModelConfig(
        summary_dim=8,
        context_dim=8,
        edge_dim=8,
        query_dim=8,
        d_model=16,
        num_heads=4,
        num_blocks=1,
        path_rows=3,
        evidence_rows=3,
        dropout=0.0,
        use_candidate_evidence_null=candidate_null,
    )


def test_candidate_null_is_exchangeable_finite_and_differentiable() -> None:
    torch.manual_seed(41)
    decoder = CandidateEvidenceNullDecoder(d_model=8, control_width=6)
    outputs = _candidate_outputs(5)
    owners = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)

    logits = decoder(
        outputs,
        owners,
        graph_count=3,
    )
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted_outputs = CandidateOutputs(
        *(value[permutation] for value in outputs.tensors())
    )
    replay = decoder(
        permuted_outputs,
        owners[permutation],
        graph_count=3,
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits, replay, atol=1e-6, rtol=1e-6)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in decoder.parameters())
    assert not any(
        "position" in name.lower() or "row_index" in name.lower()
        for name, _ in decoder.named_parameters()
    )


def test_candidate_null_handles_an_empty_candidate_set() -> None:
    decoder = CandidateEvidenceNullDecoder(d_model=8, control_width=6)
    empty = _candidate_outputs(0)

    logits = decoder(
        empty,
        torch.empty(0, dtype=torch.int32),
        graph_count=2,
    )

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_candidate_null_policy_uses_raw_zero_margin_and_allows_empty() -> None:
    batch = _batch()
    model = PooledScorer(_model_config()).eval()
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=5,
            frontier_width=8,
            hypotheses_per_node=2,
            context_read_budget=4,
            evidence_selection_budget=8,
            search_budget=128,
            max_depth=6,
            evidence_threshold=0.0,
            evidence_selection_policy="candidate_null",
        )
    )
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(model, batch, hypotheses, evidence, state)
    count = proposal.expansion.total_arcs
    evidence_logits = torch.linspace(-2.0, 2.0, count)
    proposal = replace(
        proposal,
        candidate_outputs=replace(
            proposal.candidate_outputs,
            evidence_logits=evidence_logits,
        ),
        evidence_candidate_null_logits=torch.tensor([0.25]),
    )

    selected = controller.choose_actions(
        proposal,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(3),
    ).evidence_candidate_indices
    assert selected.tolist() == torch.nonzero(
        evidence_logits > 0.25,
        as_tuple=False,
    ).flatten().flip(0).tolist()

    empty = controller.choose_actions(
        replace(
            proposal,
            evidence_candidate_null_logits=torch.tensor([3.0]),
        ),
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(3),
    ).evidence_candidate_indices
    assert empty.numel() == 0


def test_candidate_null_is_recomputed_after_context_refinement() -> None:
    batch = _batch()
    model = PooledScorer(_model_config()).eval()
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=5,
            frontier_width=8,
            hypotheses_per_node=2,
            context_read_budget=4,
            evidence_selection_budget=8,
            search_budget=128,
            max_depth=6,
            context_threshold=0.0,
            evidence_selection_policy="candidate_null",
        )
    )
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(model, batch, hypotheses, evidence, state)

    selection = controller.select_actions(
        model,
        batch,
        proposal,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(7),
    )
    expected = model.candidate_evidence_null_logits(
        selection.proposal.candidate_outputs,
        selection.proposal.candidate_graph_ids,
        graph_count=batch.graph_count,
    )

    assert selection.proposal.context_refined
    assert selection.proposal.evidence_candidate_null_logits is not None
    assert torch.equal(
        selection.proposal.evidence_candidate_null_logits,
        expected,
    )


def test_candidate_null_policy_requires_the_candidate_decoder(
    tmp_path: Path,
) -> None:
    config = json.loads(Path("configs/spider_v0_5/X0.json").read_text())
    config["controller"]["evidence_selection_policy"] = "candidate_null"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="candidate evidence null"):
        load_experiment(path)


def test_v0_6_partitions_have_disjoint_symbol_namespaces() -> None:
    specs = default_zero_shot_specs()

    assert [spec.case_count for spec in specs] == [8192, 512, 512, 1024]
    assert all(spec.dataset_version == V06_DATASET_VERSION for spec in specs)
    assert all(not spec.sealed for spec in specs)

    seen: set[str] = set()
    for spec in specs:
        cases = generate_zero_shot_cases(spec, limit=128)
        symbols = observable_symbols(cases)
        assert symbols
        assert seen.isdisjoint(symbols)
        seen.update(symbols)


def test_v0_6_configs_use_no_calibrated_threshold_or_count() -> None:
    for arm in ("Z0", "Z1"):
        raw = json.loads(
            (ROOT / f"configs/spider_v0_6/{arm}.json").read_text()
        )
        assert raw["dataset"]["version"] == V06_DATASET_VERSION
        assert raw["dataset"]["fit_operating_policy"] is False
        assert raw["model"]["use_evidence_cardinality"] is False
        assert raw["model"]["use_candidate_evidence_count"] is False
        assert raw["controller"]["evidence_selection_policy"] in {
            "learned_null",
            "candidate_null",
        }

    z1 = load_experiment(ROOT / "configs/spider_v0_6/Z1.json")
    assert z1.model_config.use_candidate_evidence_null
    assert z1.controller_config.evidence_selection_policy == "candidate_null"

    validate_calibration_source(
        split_name="model_selection",
        dataset_version=V06_DATASET_VERSION,
    )
