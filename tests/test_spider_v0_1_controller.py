from __future__ import annotations

from dataclasses import replace
import random

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
    ActionSource,
    ControllerActions,
    ControllerState,
    SparseControllerConfig,
    SparseWavefrontController,
    SpiderModel,
    SpiderModelConfig,
    StateOracle,
    candidate_control_features,
    termination_control_features,
)
from hippocampus.spider.types import CandidateOutputs


def _fixture(
    family: ProgramFamily = ProgramFamily.REACHABILITY,
    *,
    path_length: int = 2,
):
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    case = GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=8,
            max_nodes=10,
            min_path_length=path_length,
            max_path_length=path_length,
        )
    ).generate(
        family=family,
        seed=222,
        answerable=True,
        require_multiple_paths=family is ProgramFamily.REACHABILITY,
    )
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=31)
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
            dropout=0.0,
        )
    ).eval()
    config = SparseControllerConfig(
        max_rounds=5,
        frontier_width=6,
        hypotheses_per_node=2,
        context_read_budget=4,
        evidence_selection_budget=4,
        search_budget=64,
        max_depth=6,
        expand_threshold=0.5,
    )
    return case, batch, model, SparseWavefrontController(config)


def _constant_outputs(
    outputs: CandidateOutputs,
    *,
    expand: float | None = None,
    context: float | None = None,
    evidence: float | None = None,
) -> CandidateOutputs:
    def value(source: torch.Tensor, replacement: float | None) -> torch.Tensor:
        return (
            source
            if replacement is None
            else torch.full_like(source, replacement)
        )

    return CandidateOutputs(
        next_path_state=outputs.next_path_state,
        priority_logits=outputs.priority_logits,
        expand_logits=value(outputs.expand_logits, expand),
        context_logits=value(outputs.context_logits, context),
        evidence_logits=value(outputs.evidence_logits, evidence),
        remaining_cost=outputs.remaining_cost,
        support_logits=outputs.support_logits,
        conflict_logits=outputs.conflict_logits,
    )


def test_candidate_controls_are_identical_in_training_and_runtime() -> None:
    _, batch, model, controller = _fixture(ProgramFamily.LATEST_VALID)
    hypotheses = model.initial_hypotheses(batch)
    state = ControllerState.initial()
    proposal = controller.propose(
        model,
        batch,
        hypotheses,
        model.initial_evidence(batch),
        state,
    )
    independently_built = candidate_control_features(
        hypotheses,
        proposal.expansion,
        state,
        config=controller.config,
        search_limit=proposal.search_limit,
        context_limit=proposal.context_limit,
        dtype=hypotheses.path_state.dtype,
    )

    assert torch.equal(
        proposal.candidate_control_features,
        independently_built,
    )
    assert proposal.candidate_control_features[:, 2].gt(0).all()
    assert proposal.candidate_control_features[:, 3].gt(0).all()


def test_termination_controls_are_byte_equal_across_execution_modes() -> None:
    _, batch, model, controller = _fixture()
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    proposal = controller.propose(
        model,
        batch,
        hypotheses,
        evidence,
        ControllerState.initial(),
    )
    actions = ControllerActions.empty(batch.device)
    transition = controller.apply(
        model,
        batch,
        hypotheses,
        evidence,
        ControllerState.initial(),
        proposal,
        actions,
    )
    rebuilt = termination_control_features(
        batch,
        transition.next_hypotheses,
        transition.next_controller_state,
        config=controller.config,
        search_limit=proposal.search_limit,
        context_limit=proposal.context_limit,
    )

    assert torch.equal(transition.termination_control, rebuilt)
    assert transition.termination_control[:, 0].item() == pytest.approx(0.2)
    assert transition.termination_control[:, 3].item() == 1.0


def test_model_context_action_changes_the_shared_transition() -> None:
    _, batch, model, controller = _fixture(ProgramFamily.LATEST_VALID)
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(model, batch, hypotheses, evidence, state)
    proposal = replace(
        proposal,
        candidate_outputs=_constant_outputs(
            proposal.candidate_outputs,
            expand=-100.0,
            context=100.0,
            evidence=-100.0,
        ),
    )
    actions = controller.choose_actions(
        proposal,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(3),
    )
    transition = controller.apply(
        model,
        batch,
        hypotheses,
        evidence,
        state,
        proposal,
        actions,
    )

    assert actions.context_source is ActionSource.MODEL
    assert actions.context_candidate_indices.numel() > 0
    assert transition.next_controller_state.contexts_read > 0
    assert transition.next_controller_state.context_ledger


def test_model_evidence_action_changes_the_shared_transition() -> None:
    _, batch, model, controller = _fixture(ProgramFamily.LOOKUP, path_length=1)
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(model, batch, hypotheses, evidence, state)
    proposal = replace(
        proposal,
        candidate_outputs=_constant_outputs(
            proposal.candidate_outputs,
            expand=-100.0,
            context=-100.0,
            evidence=100.0,
        ),
    )
    actions = controller.choose_actions(
        proposal,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(4),
    )
    transition = controller.apply(
        model,
        batch,
        hypotheses,
        evidence,
        state,
        proposal,
        actions,
    )

    assert actions.evidence_source is ActionSource.MODEL
    assert actions.evidence_candidate_indices.numel() > 0
    assert transition.next_controller_state.evidence_ledger
    assert not torch.equal(transition.next_evidence, evidence)


def test_evidence_can_be_included_without_frontier_expansion() -> None:
    _, batch, model, controller = _fixture(ProgramFamily.LOOKUP, path_length=1)
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(model, batch, hypotheses, evidence, state)
    actions = ControllerActions(
        frontier_candidate_indices=torch.empty(
            0, dtype=torch.int64, device=batch.device
        ),
        context_candidate_indices=torch.empty(
            0, dtype=torch.int64, device=batch.device
        ),
        evidence_candidate_indices=torch.tensor(
            [0], dtype=torch.int64, device=batch.device
        ),
        frontier_source=ActionSource.MODEL,
        context_source=ActionSource.MODEL,
        evidence_source=ActionSource.MODEL,
        termination_source=ActionSource.MODEL,
    )
    transition = controller.apply(
        model,
        batch,
        hypotheses,
        evidence,
        state,
        proposal,
        actions,
    )

    assert transition.next_hypotheses.count == 0
    assert len(transition.next_controller_state.trace_ledger) == 0
    assert len(transition.next_controller_state.evidence_ledger) == 1
    assert not torch.equal(transition.next_evidence, evidence)


def test_model_policy_can_choose_an_empty_frontier() -> None:
    _, batch, model, controller = _fixture()
    hypotheses = model.initial_hypotheses(batch)
    state = ControllerState.initial()
    proposal = controller.propose(
        model,
        batch,
        hypotheses,
        model.initial_evidence(batch),
        state,
    )
    proposal = replace(
        proposal,
        candidate_outputs=_constant_outputs(
            proposal.candidate_outputs,
            expand=-100.0,
            context=-100.0,
            evidence=-100.0,
        ),
    )
    actions = controller.choose_actions(
        proposal,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(5),
    )

    assert actions.frontier_candidate_indices.numel() == 0


def test_oracle_selected_actions_follow_independent_schedule() -> None:
    case, batch, model, controller = _fixture(ProgramFamily.LATEST_VALID)
    hypotheses = model.initial_hypotheses(batch)
    state = ControllerState.initial()
    proposal = controller.propose(
        model,
        batch,
        hypotheses,
        model.initial_evidence(batch),
        state,
    )
    supervision = StateOracle(case, batch, controller.config).label(
        proposal,
        hypotheses,
        state,
    )
    actions = controller.choose_actions(
        proposal,
        supervision=supervision,
        state=state,
        schedule=ActionSchedule(
            frontier=1.0,
            context=0.0,
            evidence=1.0,
            termination=0.0,
        ),
        randomizer=random.Random(1),
    )

    assert actions.frontier_source is ActionSource.ORACLE
    assert actions.context_source is ActionSource.MODEL
    assert actions.evidence_source is ActionSource.ORACLE
    assert actions.termination_source is ActionSource.MODEL


def test_state_oracle_continues_recoverable_off_oracle_frontier() -> None:
    case, batch, model, controller = _fixture()
    oracle = StateOracle(case, batch, controller.config)
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(model, batch, hypotheses, evidence, state)
    supervision = oracle.label(proposal, hypotheses, state)
    acceptable = torch.nonzero(
        supervision.candidates.acceptable,
        as_tuple=False,
    ).flatten()
    assert acceptable.numel() >= 2
    # Keep one of multiple legal paths: this no longer equals the recorded
    # parallel frontier but still has a valid completion.
    actions = ControllerActions(
        frontier_candidate_indices=acceptable[:1],
        context_candidate_indices=torch.empty(
            0, dtype=torch.int64, device=batch.device
        ),
        evidence_candidate_indices=torch.empty(
            0, dtype=torch.int64, device=batch.device
        ),
        frontier_source=ActionSource.MODEL,
        context_source=ActionSource.MODEL,
        evidence_source=ActionSource.MODEL,
        termination_source=ActionSource.MODEL,
    )
    transition = controller.apply(
        model,
        batch,
        hypotheses,
        evidence,
        state,
        proposal,
        actions,
    )

    assert oracle.termination_target(transition).decision.value == "continue"


def test_state_oracle_handles_partial_and_duplicate_frontiers() -> None:
    case, batch, model, controller = _fixture()
    oracle = StateOracle(case, batch, controller.config)
    hypotheses = model.initial_hypotheses(batch).repeat_occurrences(
        torch.tensor([0, 0])
    )
    state = ControllerState.initial()
    proposal = controller.propose(
        model,
        batch,
        hypotheses,
        model.initial_evidence(batch),
        state,
    )
    supervision = oracle.label(proposal, hypotheses, state)

    assert supervision.recoverable
    assert supervision.candidates.acceptable.sum().item() >= 2
    half = proposal.expansion.total_arcs // 2
    assert torch.equal(
        supervision.candidates.acceptable[:half],
        supervision.candidates.acceptable[half:],
    )


def test_state_oracle_marks_unrecoverable_budget_state_incomplete() -> None:
    case, batch, model, _ = _fixture()
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=5,
            frontier_width=4,
            hypotheses_per_node=2,
            context_read_budget=2,
            evidence_selection_budget=2,
            search_budget=0,
            max_depth=5,
        )
    )
    oracle = StateOracle(case, batch, controller.config)
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(model, batch, hypotheses, evidence, state)
    transition = controller.apply(
        model,
        batch,
        hypotheses,
        evidence,
        state,
        proposal,
        ControllerActions.empty(batch.device),
    )

    assert (
        oracle.termination_target(transition).decision.value
        == "unknown_incomplete"
    )
