from __future__ import annotations

from dataclasses import replace
import json
import random
from pathlib import Path

import pytest
import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from hippocampus.spider import (
    ActionSchedule,
    ControllerState,
    SparseControllerConfig,
    SparseWavefrontController,
    SpiderLossConfig,
    SpiderModel,
    SpiderModelConfig,
    evidence_cardinality_loss_term,
    evidence_null_loss_term,
    load_experiment,
)
from hippocampus.spider.types import CandidateOutputs


def _fixture(policy: str):
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
        seed=414,
        answerable=True,
        require_multiple_paths=True,
    )
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=51)
    batch = pack_rendered_cases(
        (case,),
        (renderer.render(case),),
        schema=schema,
    )
    model = SpiderModel(
        SpiderModelConfig(
            summary_dim=8,
            context_dim=8,
            edge_dim=8,
            query_dim=8,
            d_model=16,
            num_heads=4,
            num_blocks=1,
            path_rows=3,
            evidence_rows=3,
            use_evidence_null=policy in {"learned_null", "null_cardinality"},
            use_evidence_cardinality=policy in {
                "cardinality",
                "null_cardinality",
            },
        )
    ).eval()
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=5,
            frontier_width=8,
            hypotheses_per_node=2,
            context_read_budget=4,
            evidence_selection_budget=8,
            search_budget=128,
            max_depth=6,
            evidence_selection_policy=policy,
        )
    )
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(model, batch, hypotheses, evidence, state)
    assert proposal.expansion.total_arcs >= 3
    logits = torch.arange(
        proposal.expansion.total_arcs,
        dtype=proposal.candidate_outputs.evidence_logits.dtype,
        device=batch.device,
    )
    outputs = proposal.candidate_outputs
    proposal = replace(
        proposal,
        candidate_outputs=CandidateOutputs(
            next_path_state=outputs.next_path_state,
            priority_logits=outputs.priority_logits,
            expand_logits=outputs.expand_logits,
            context_logits=outputs.context_logits,
            evidence_logits=logits,
            remaining_cost=outputs.remaining_cost,
            support_logits=outputs.support_logits,
            conflict_logits=outputs.conflict_logits,
        ),
    )
    return batch, model, controller, proposal, state


def _choose(controller, proposal, state):
    return controller.choose_actions(
        proposal,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(5),
    ).evidence_candidate_indices


def test_learned_null_selects_only_candidates_above_graph_boundary() -> None:
    _, _, controller, proposal, state = _fixture("learned_null")
    count = proposal.expansion.total_arcs
    proposal = replace(
        proposal,
        evidence_null_logits=torch.tensor([count - 1.5]),
    )

    selected = _choose(controller, proposal, state)

    assert selected.tolist() == [count - 1]
    empty = _choose(
        controller,
        replace(proposal, evidence_null_logits=torch.tensor([count + 1.0])),
        state,
    )
    assert empty.numel() == 0


def test_cardinality_policy_uses_total_count_and_stable_top_k() -> None:
    _, _, controller, proposal, state = _fixture("cardinality")
    count = proposal.expansion.total_arcs
    cardinality = torch.full((1, 5), -10.0)
    cardinality[0, 2] = 10.0
    proposal = replace(proposal, evidence_cardinality_logits=cardinality)

    selected = _choose(controller, proposal, state)

    assert selected.tolist() == [count - 1, count - 2]
    after_one_recorded = _choose(
        controller,
        replace(
            proposal,
            evidence_selected_by_graph=torch.tensor([1]),
        ),
        state,
    )
    assert after_one_recorded.tolist() == [count - 1]

    zero = cardinality.clone()
    zero[0, 2] = -10.0
    zero[0, 0] = 10.0
    assert _choose(
        controller,
        replace(proposal, evidence_cardinality_logits=zero),
        state,
    ).numel() == 0


def test_null_cardinality_combines_boundary_and_count() -> None:
    _, _, controller, proposal, state = _fixture("null_cardinality")
    count = proposal.expansion.total_arcs
    cardinality = torch.full((1, 5), -10.0)
    cardinality[0, 2] = 10.0
    proposal = replace(
        proposal,
        evidence_null_logits=torch.tensor([count - 1.5]),
        evidence_cardinality_logits=cardinality,
    )

    assert _choose(controller, proposal, state).tolist() == [count - 1]


def test_selection_heads_have_graph_shapes_and_receive_gradients() -> None:
    batch, model, _, proposal, _ = _fixture("null_cardinality")

    assert proposal.evidence_null_logits is not None
    assert proposal.evidence_null_logits.shape == (batch.graph_count,)
    assert proposal.evidence_cardinality_logits is not None
    assert proposal.evidence_cardinality_logits.shape == (batch.graph_count, 5)
    (proposal.evidence_null_logits.sum() + proposal.evidence_cardinality_logits.sum()).backward()
    assert model.evidence_null_head[-1].weight.grad is not None
    assert model.evidence_cardinality_head[-1].weight.grad is not None


def test_set_decoder_losses_train_relative_boundary_and_four_plus_class() -> None:
    null = torch.tensor([0.0], requires_grad=True)
    candidates = torch.tensor([1.0, -1.0], requires_grad=True)
    config = SpiderLossConfig(evidence_null=0.7, evidence_cardinality=0.8)
    null_term = evidence_null_loss_term(
        null,
        candidates,
        torch.tensor([True, False]),
        torch.tensor([0, 0]),
        config=config,
    )
    cardinality = torch.zeros((1, 5), requires_grad=True)
    cardinality_term = evidence_cardinality_loss_term(
        cardinality,
        torch.tensor([9]),
        config=config,
    )

    assert null_term is not None
    assert null_term.target_count == 2
    assert cardinality_term is not None
    assert cardinality_term.target_count == 1
    (null_term.weighted + cardinality_term.weighted).backward()
    assert null.grad is not None
    assert candidates.grad is not None
    assert cardinality.grad is not None
    assert cardinality.grad[0, 4] < 0


@pytest.mark.parametrize(
    ("policy", "null", "cardinality", "message"),
    (
        ("learned_null", False, False, "evidence null"),
        ("cardinality", False, False, "cardinality"),
        ("null_cardinality", True, False, "cardinality"),
    ),
)
def test_experiment_config_rejects_missing_policy_head(
    tmp_path: Path,
    policy: str,
    null: bool,
    cardinality: bool,
    message: str,
) -> None:
    source = Path("configs/spider_v0_4/phase_d_D0.json")
    config = json.loads(source.read_text())
    config["controller"]["evidence_selection_policy"] = policy
    config["model"]["use_evidence_null"] = null
    config["model"]["use_evidence_cardinality"] = cardinality
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        load_experiment(path)
