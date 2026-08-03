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
    V05_DATASET_VERSION,
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    audit_aligned_program_labels,
    build_aligned_dev_manifest,
    default_score_decode_specs,
    generate_score_decode_cases,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ActionSchedule,
    CandidateEvidenceSetDecoder,
    ControllerState,
    PooledScorer,
    SparseControllerConfig,
    SparseWavefrontController,
    SpiderLossConfig,
    SpiderModelConfig,
    candidate_evidence_count_targets,
    evidence_candidate_count_loss_term,
    load_experiment,
)
from hippocampus.spider.types import CandidateOutputs


ROOT = Path(__file__).resolve().parents[1]


def _autoresearch_module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = ROOT / "scripts/run_spider_v0_5_autoresearch.py"
    spec = importlib.util.spec_from_file_location("spider_v05_autoresearch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        seed=913,
        answerable=True,
        require_multiple_paths=True,
    )
    renderer = SyntheticManifoldRenderer(
        schema,
        query_dim=8,
        seed=177,
        geometry="orthogonal_aligned",
    )
    return pack_rendered_cases(
        (case,),
        (renderer.render(case, row_permutation_seed=row_seed),),
        schema=schema,
    )


def _config(
    *,
    readout: str = "shared",
    candidate_count: bool = False,
) -> SpiderModelConfig:
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
        evidence_readout=readout,
        use_candidate_evidence_count=candidate_count,
    )


def _candidate_outputs(count: int, *, d_model: int = 8) -> CandidateOutputs:
    path = torch.randn(count, 3, d_model, requires_grad=True)
    logits = [
        torch.randn(count, requires_grad=True)
        for _ in range(7)
    ]
    return CandidateOutputs(path, *logits)


def test_pairwise_matcher_is_exchangeable_and_receives_gradients() -> None:
    torch.manual_seed(23)
    base = _batch(row_seed=1)
    permuted = _batch(row_seed=999)
    model = PooledScorer(_config(readout="pairwise_matcher")).eval()

    def score(batch):
        hypotheses = model.initial_hypotheses(batch)
        expansion = batch.graph.expand_frontier(hypotheses.node_ids)
        return model.score_candidates(
            batch,
            hypotheses,
            expansion,
            model.initial_evidence(batch),
        ).evidence_logits

    base_logits = score(base)
    permuted_logits = score(permuted)

    assert torch.allclose(base_logits, permuted_logits, atol=1e-6, rtol=1e-6)
    base_logits.sum().backward()
    assert model.evidence_readout is not None
    assert all(
        parameter.grad is not None
        for parameter in model.evidence_readout.parameters()
    )
    assert not any(
        "position" in name.lower() or "row_index" in name.lower()
        for name, _ in model.evidence_readout.named_parameters()
    )


def test_candidate_set_decoder_is_permutation_invariant_and_handles_empty_graphs() -> None:
    torch.manual_seed(29)
    decoder = CandidateEvidenceSetDecoder(d_model=8)
    outputs = _candidate_outputs(4)
    owners = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    logits = decoder(outputs, owners, graph_count=3)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = CandidateOutputs(
        *(value[permutation] for value in outputs.tensors())
    )
    permuted_logits = decoder(
        permuted,
        owners[permutation],
        graph_count=3,
    )

    assert logits.shape == (3, 5)
    assert torch.allclose(logits, permuted_logits, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in decoder.parameters())

    empty = _candidate_outputs(0)
    empty_logits = decoder(
        empty,
        torch.empty(0, dtype=torch.int32),
        graph_count=2,
    )
    assert empty_logits.shape == (2, 5)
    assert torch.isfinite(empty_logits).all()


def test_candidate_count_targets_count_unique_current_positive_edges() -> None:
    targets = candidate_evidence_count_targets(
        torch.tensor([True, True, False, True, True, True, True]),
        torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int32),
        torch.tensor([4, 4, 5, 7, 8, 9, 10], dtype=torch.int32),
        graph_count=3,
    )

    assert targets.tolist() == [1, 4, 0]


def test_candidate_count_loss_trains_zero_and_four_plus_classes() -> None:
    logits = torch.zeros((2, 5), requires_grad=True)
    term = evidence_candidate_count_loss_term(
        logits,
        torch.tensor([0, 9]),
        config=SpiderLossConfig(evidence_candidate_count=0.75),
    )

    assert term is not None
    assert term.target_count == 2
    term.weighted.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0
    assert logits.grad[1, 4] < 0


def test_candidate_count_policy_uses_current_count_without_ledger_subtraction() -> None:
    batch = _batch()
    model = PooledScorer(_config(candidate_count=True)).eval()
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=5,
            frontier_width=8,
            hypotheses_per_node=2,
            context_read_budget=4,
            evidence_selection_budget=8,
            search_budget=128,
            max_depth=6,
            evidence_selection_policy="candidate_count",
        )
    )
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(model, batch, hypotheses, evidence, state)
    count = proposal.expansion.total_arcs
    outputs = proposal.candidate_outputs
    ranked = torch.arange(count, dtype=outputs.evidence_logits.dtype)
    count_logits = torch.full((1, 5), -10.0)
    count_logits[0, 2] = 10.0
    proposal = replace(
        proposal,
        candidate_outputs=replace(outputs, evidence_logits=ranked),
        evidence_candidate_count_logits=count_logits,
        evidence_selected_by_graph=torch.tensor([3]),
    )

    selected = controller.choose_actions(
        proposal,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(5),
    ).evidence_candidate_indices

    assert selected.tolist() == [count - 1, count - 2]
    zero_logits = count_logits.clone()
    zero_logits[0, 2] = -10.0
    zero_logits[0, 0] = 10.0
    empty = controller.choose_actions(
        replace(proposal, evidence_candidate_count_logits=zero_logits),
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(5),
    ).evidence_candidate_indices
    assert empty.numel() == 0


def test_candidate_decoder_is_recomputed_after_context_refinement() -> None:
    batch = _batch()
    model = PooledScorer(_config(candidate_count=True)).eval()
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
            evidence_selection_policy="candidate_count",
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
    expected = model.candidate_evidence_count_logits(
        selection.proposal.candidate_outputs,
        selection.proposal.candidate_graph_ids,
        graph_count=batch.graph_count,
    )

    assert selection.proposal.context_refined
    assert selection.proposal.evidence_candidate_count_logits is not None
    assert torch.equal(
        selection.proposal.evidence_candidate_count_logits,
        expected,
    )


def test_experiment_rejects_candidate_count_policy_without_decoder(
    tmp_path: Path,
) -> None:
    config = json.loads(
        Path("configs/spider_v0_4/phase_f_F0.json").read_text()
    )
    config["controller"]["evidence_selection_policy"] = "candidate_count"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="candidate evidence count"):
        load_experiment(path)


def test_v0_5_partitions_are_new_balanced_and_non_sealed() -> None:
    specs = default_score_decode_specs()

    assert [spec.case_count for spec in specs] == [8192, 512, 512, 1024]
    assert all(spec.dataset_version == V05_DATASET_VERSION for spec in specs)
    assert all(not spec.sealed for spec in specs)
    assert min(spec.seed_start for spec in specs) >= 810_000

    seen_case_ids: set[str] = set()
    seen_base_ids: set[str] = set()
    for spec in specs:
        cases = generate_score_decode_cases(spec, limit=128)
        manifest = build_aligned_dev_manifest(spec, cases)
        audit = audit_aligned_program_labels(cases)

        assert not seen_case_ids.intersection(manifest.case_ids)
        assert not seen_base_ids.intersection(manifest.base_case_ids)
        assert audit.invalid_case_count == 0
        assert audit.evidence_label_mismatch_count == 0
        assert audit.unsupported_case_count == 0
        assert audit.query_cardinality_answerability_accuracy == (
            audit.query_cardinality_majority_accuracy
        )
        assert manifest.distributions["family"] == {
            "corroboration": 32,
            "latest_valid": 32,
            "lookup": 32,
            "reachability": 32,
        }
        assert manifest.distributions["outcome"] == {
            "answerable": 64,
            "unknown": 64,
        }
        seen_case_ids.update(manifest.case_ids)
        seen_base_ids.update(manifest.base_case_ids)


def test_v0_5_configs_form_only_the_registered_factorial() -> None:
    configs = {
        arm: json.loads(Path(f"configs/spider_v0_5/{arm}.json").read_text())
        for arm in ("X0", "X1", "X2", "X3")
    }
    manifest = json.loads(
        Path("artifacts/spider_v0_5/splits/MANIFEST_INDEX.json").read_text()
    )
    for config in configs.values():
        assert config["dataset"]["protocol"] == "spider-v0.5-score-decode"
        assert config["dataset"]["aggregate_sha256"] == (
            manifest["aggregate_sha256"]
        )
        assert config["dataset"]["training_case_count"] == 8192
        assert config["training"]["steps"] == 2000

    assert configs["X0"]["model"]["evidence_readout"] == "shared"
    assert configs["X1"]["model"]["evidence_readout"] == "pairwise_matcher"
    assert configs["X2"]["model"]["evidence_readout"] == "shared"
    assert configs["X3"]["model"]["evidence_readout"] == "pairwise_matcher"
    assert configs["X0"]["controller"]["evidence_selection_policy"] == (
        "threshold"
    )
    assert configs["X1"]["controller"]["evidence_selection_policy"] == (
        "threshold"
    )
    for arm in ("X2", "X3"):
        assert configs[arm]["controller"]["evidence_selection_policy"] == (
            "candidate_count"
        )
        assert configs[arm]["model"]["use_candidate_evidence_count"]
        assert configs[arm]["loss"]["evidence_candidate_count"] == 1.0

    common_model = {
        key: value
        for key, value in configs["X0"]["model"].items()
        if key not in {"evidence_readout", "use_candidate_evidence_count"}
    }
    for config in configs.values():
        assert {
            key: value
            for key, value in config["model"].items()
            if key not in {"evidence_readout", "use_candidate_evidence_count"}
        } == common_model


@pytest.mark.parametrize("arm", ("X0", "X1", "X2", "X3"))
def test_v0_5_configs_resolve(arm: str) -> None:
    experiment = load_experiment(f"configs/spider_v0_5/{arm}.json")

    assert experiment.raw["dataset"]["version"] == V05_DATASET_VERSION
    assert experiment.device.type == "cuda"


def _gate_metrics(
    *,
    exact: float,
    precision: float,
    recall: float,
    coverage: float,
    lookup: float,
    latest: float,
    corroboration: float,
) -> dict[str, object]:
    return {
        "primary_metric": {
            "exact_evidence_set_accuracy": exact,
            "precision": precision,
            "recall": recall,
            "scored_positive_coverage": coverage,
        },
        "per_family": {
            "lookup": {"recall": lookup},
            "latest_valid": {"exact_evidence_set_accuracy": latest},
            "corroboration": {
                "exact_evidence_set_accuracy": corroboration
            },
        },
    }


def test_v0_5_gate_requires_recall_precision_and_family_safety() -> None:
    module = _autoresearch_module()
    control = _gate_metrics(
        exact=0.70,
        precision=0.90,
        recall=0.60,
        coverage=0.99,
        lookup=0.10,
        latest=0.95,
        corroboration=0.94,
    )
    passing = _gate_metrics(
        exact=0.75,
        precision=0.91,
        recall=0.65,
        coverage=0.99,
        lookup=0.30,
        latest=0.93,
        corroboration=0.92,
    )

    assert module._seed_gate(control, passing)["advances"]
    for override in (
        {"exact": 0.749},
        {"precision": 0.899},
        {"recall": 0.649},
        {"coverage": 0.979},
        {"lookup": 0.299},
        {"latest": 0.929},
        {"corroboration": 0.919},
    ):
        values = {
            "exact": 0.75,
            "precision": 0.91,
            "recall": 0.65,
            "coverage": 0.99,
            "lookup": 0.30,
            "latest": 0.93,
            "corroboration": 0.92,
            **override,
        }
        assert not module._seed_gate(
            control,
            _gate_metrics(**values),
        )["advances"]


def test_v0_5_orchestrator_lock_and_resume_stages(tmp_path: Path) -> None:
    module = _autoresearch_module()
    with module._campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another v0.5 orchestrator"):
            with module._campaign_lock(tmp_path):
                pass

    run = tmp_path / "run"
    run.mkdir()
    partial = run / "checkpoint_step_000750.pt"
    partial.touch()
    assert module._interrupted_stage(run) == ("training", partial)
    (run / "checkpoint.pt").touch()
    assert module._interrupted_stage(run) == ("selection", None)
    (run / "evaluation_pause.json").write_text(
        json.dumps({"training_source_commit": "abc"})
    )
    assert module._interrupted_stage(run) == ("evaluation", None)
