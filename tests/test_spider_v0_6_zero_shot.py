from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import random
from pathlib import Path
import sys

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
    SpiderLossConfig,
    SpiderModelConfig,
    evidence_null_loss_term,
    evidence_null_margin_loss_term,
    load_experiment,
)
from hippocampus.spider.types import CandidateOutputs
from hippocampus.spider.calibration import validate_calibration_source


ROOT = Path(__file__).resolve().parents[1]


def _autoresearch_module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = ROOT / "scripts/run_spider_v0_6_autoresearch.py"
    spec = importlib.util.spec_from_file_location("spider_v06_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_graph_balanced_null_loss_ignores_duplicate_negative_mass() -> None:
    null = torch.tensor([0.3], requires_grad=True)
    logits = torch.tensor([-0.2, 2.0], requires_grad=True)
    targets = torch.tensor([True, False])
    owners = torch.tensor([0, 0], dtype=torch.int32)
    config = SpiderLossConfig(
        evidence_null=1.0,
        evidence_null_mode="graph_balanced",
    )

    original = evidence_null_loss_term(
        null,
        logits,
        targets,
        owners,
        config=config,
    )
    duplicated = evidence_null_loss_term(
        null,
        torch.stack((logits[0], logits[1], logits[1])),
        torch.tensor([True, False, False]),
        torch.tensor([0, 0, 0], dtype=torch.int32),
        config=config,
    )

    assert original is not None and duplicated is not None
    assert original.target_count == 1
    assert duplicated.target_count == 1
    assert torch.allclose(original.raw, duplicated.raw)
    original.weighted.backward()
    assert null.grad is not None
    assert logits.grad is not None


def test_plain_null_loss_remains_candidate_weighted() -> None:
    null = torch.tensor([0.3])
    logits = torch.tensor([-0.2, 2.0])
    targets = torch.tensor([True, False])
    owners = torch.tensor([0, 0], dtype=torch.int32)
    config = SpiderLossConfig(
        evidence_null=1.0,
        evidence_null_mode="plain",
    )

    original = evidence_null_loss_term(
        null,
        logits,
        targets,
        owners,
        config=config,
    )
    duplicated = evidence_null_loss_term(
        null,
        torch.tensor([-0.2, 2.0, 2.0]),
        torch.tensor([True, False, False]),
        torch.tensor([0, 0, 0], dtype=torch.int32),
        config=config,
    )

    assert original is not None and duplicated is not None
    assert original.target_count == 2
    assert duplicated.target_count == 3
    assert not torch.allclose(original.raw, duplicated.raw)


def test_graph_balanced_null_loss_handles_all_negative_graphs() -> None:
    null = torch.tensor([0.1, -0.2], requires_grad=True)
    logits = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
    term = evidence_null_loss_term(
        null,
        logits,
        torch.tensor([False, False, True]),
        torch.tensor([0, 0, 1], dtype=torch.int32),
        config=SpiderLossConfig(
            evidence_null=1.0,
            evidence_null_mode="graph_balanced",
        ),
    )

    assert term is not None
    assert term.target_count == 2
    assert torch.isfinite(term.raw)
    term.weighted.backward()
    assert torch.isfinite(null.grad).all()
    assert torch.isfinite(logits.grad).all()


def test_null_margin_uses_only_bounded_plausible_hard_negatives() -> None:
    null = torch.tensor([0.5], requires_grad=True)
    logits = torch.tensor([0.5, 3.0, 1.5, -2.0], requires_grad=True)
    targets = torch.tensor([True, False, False, False])
    plausible = torch.tensor([False, False, True, True])
    term = evidence_null_margin_loss_term(
        null,
        logits,
        targets,
        plausible,
        torch.zeros(4, dtype=torch.int32),
        config=SpiderLossConfig(
            evidence_null_margin=0.25,
            evidence_null_margin_value=0.2,
            evidence_null_hard_negative_count=1,
        ),
    )

    assert term is not None
    expected = 0.5 * (
        torch.nn.functional.softplus(torch.tensor(0.2))
        + torch.nn.functional.softplus(torch.tensor(1.2))
    )
    assert term.target_count == 1
    assert torch.allclose(term.raw, expected)
    term.weighted.backward()
    assert null.grad is not None
    assert logits.grad is not None
    assert logits.grad[0] != 0
    assert logits.grad[1] == 0
    assert logits.grad[2] != 0
    assert logits.grad[3] == 0


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
    for arm in ("Z0", "Z1", "Z2", "Z3"):
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
    z2 = load_experiment(ROOT / "configs/spider_v0_6/Z2.json")
    assert z2.loss_config.evidence_null_mode == "graph_balanced"
    z3 = load_experiment(ROOT / "configs/spider_v0_6/Z3.json")
    assert z3.loss_config.evidence_null_mode == "graph_balanced"
    assert z3.loss_config.evidence_null_margin == pytest.approx(0.25)
    assert z3.loss_config.evidence_null_hard_negative_count == 4

    module = _autoresearch_module()
    assert module.ARMS == ("Z0", "Z1", "Z2", "Z3")

    validate_calibration_source(
        split_name="model_selection",
        dataset_version=V06_DATASET_VERSION,
    )


def test_v0_6_score_is_limited_by_the_weakest_required_metric() -> None:
    module = _autoresearch_module()
    row = {
        "primary_metric": {
            "exact_evidence_set_accuracy": 0.88,
            "precision": 0.91,
            "recall": 0.83,
            "scored_positive_coverage": 0.99,
        }
    }

    assert module._score(row) == pytest.approx(0.83)
    row["primary_metric"]["scored_positive_coverage"] = 0.97
    assert module._score(row) == 0.0


def test_v0_6_runner_rejects_fitted_temperature() -> None:
    module = _autoresearch_module()
    config = ROOT / "configs/spider_v0_6/Z1.json"
    metrics = {
        "experiment_id": "test",
        "config_sha256": module._sha256(config),
        "sealed_access_count": 0,
        "evidence_operating_policy": "candidate_null",
        "calibration": {
            "calibration": {
                "temperature": {
                    "accepted": True,
                    "applied_temperature": 1.2,
                }
            }
        },
        "guards": {
            "finite": True,
            "deterministic_replay_mismatches": 0,
            "row_permutation_decision_mismatches": 0,
        },
    }

    with pytest.raises(RuntimeError, match="temperature"):
        module._validate_zero_shot(metrics, arm="Z1")
