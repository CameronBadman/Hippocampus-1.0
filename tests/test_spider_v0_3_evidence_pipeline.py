from __future__ import annotations

from dataclasses import replace
import random

import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    GeneratorConfig,
    GraphProgramGenerator,
    ProgramFamily,
    SyntheticManifoldRenderer,
    generate_recurrence_necessity_pair,
    pack_rendered_cases,
)
from hippocampus.programs.schema import TerminationDecision
from hippocampus.spider import (
    ActionSchedule,
    ActionSource,
    ControllerActions,
    ControllerResult,
    ControllerRoundRecord,
    ControllerState,
    SparseControllerConfig,
    SparseWavefrontController,
    SpiderModel,
    SpiderModelConfig,
    StateOracle,
    observe_evidence_pipeline,
)
from hippocampus.spider.types import CandidateOutputs


def _model() -> SpiderModel:
    return SpiderModel(
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


def _batch(case):
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=31)
    return pack_rendered_cases(
        (case,),
        (renderer.render(case),),
        schema=schema,
    )


def _controller(*, search_budget: int = 256):
    return SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=8,
            frontier_width=8,
            hypotheses_per_node=3,
            context_read_budget=8,
            evidence_selection_budget=8,
            search_budget=search_budget,
            max_depth=10,
            expand_threshold=0.5,
            evidence_threshold=0.5,
        )
    )


class _ContextDecisiveSpider(SpiderModel):
    """Test double whose context read makes every evidence logit positive."""

    def refine_with_context(
        self,
        batch,
        expansion,
        outputs,
        context_candidate_indices,
    ) -> CandidateOutputs:
        if context_candidate_indices.numel() == 0:
            return outputs
        return replace(
            outputs,
            evidence_logits=torch.full_like(outputs.evidence_logits, 100.0),
        )


def test_proposal_retains_full_csr_enumeration_before_budget_slicing() -> None:
    case = GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=8,
            max_nodes=10,
            min_path_length=2,
            max_path_length=2,
        )
    ).generate(
        family=ProgramFamily.REACHABILITY,
        seed=222,
        answerable=True,
        require_multiple_paths=True,
    )
    batch = _batch(case)
    model = _model()
    controller = _controller(search_budget=1)

    proposal = controller.propose(
        model,
        batch,
        model.initial_hypotheses(batch),
        model.initial_evidence(batch),
        ControllerState.initial(),
    )

    assert proposal.full_expansion.total_arcs == proposal.full_arc_count
    assert proposal.full_expansion.total_arcs > proposal.expansion.total_arcs
    assert proposal.expansion.total_arcs == 1
    assert proposal.search_truncated


def test_context_refinement_precedes_evidence_and_frontier_decisions() -> None:
    case = GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=8,
            max_nodes=10,
            min_path_length=1,
            max_path_length=1,
        )
    ).generate(
        family=ProgramFamily.LATEST_VALID,
        seed=333,
        answerable=True,
    )
    batch = _batch(case)
    base = _model()
    model = _ContextDecisiveSpider(base.config).eval()
    controller = _controller()
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
        candidate_outputs=replace(
            proposal.candidate_outputs,
            context_logits=torch.full_like(
                proposal.candidate_outputs.context_logits,
                100.0,
            ),
            evidence_logits=torch.full_like(
                proposal.candidate_outputs.evidence_logits,
                -100.0,
            ),
            expand_logits=torch.full_like(
                proposal.candidate_outputs.expand_logits,
                -100.0,
            ),
        ),
    )

    selection = controller.select_actions(
        model,
        batch,
        proposal,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(0),
    )

    assert selection.actions.context_candidate_indices.numel() > 0
    assert selection.actions.evidence_candidate_indices.numel() > 0
    assert selection.proposal.context_refined
    assert selection.proposal.pre_context_outputs is not None
    assert selection.proposal.pre_context_outputs.evidence_logits.lt(0).all()
    assert selection.proposal.candidate_outputs.evidence_logits.gt(0).all()


def _final_recurrence_state(case, batch, model, controller):
    oracle = StateOracle(case, batch, controller.config)
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    for round_index in range(len(case.trace.rounds)):
        proposal = controller.propose(
            model,
            batch,
            hypotheses,
            evidence,
            state,
        )
        supervision = oracle.label(proposal, hypotheses, state)
        if round_index == len(case.trace.rounds) - 1:
            return oracle, hypotheses, evidence, state, proposal, supervision
        selection = controller.select_actions(
            model,
            batch,
            proposal,
            supervision=supervision,
            state=state,
            schedule=ActionSchedule.oracle_only(),
            randomizer=random.Random(0),
        )
        transition = controller.apply(
            model,
            batch,
            hypotheses,
            evidence,
            state,
            selection.proposal,
            selection.actions,
        )
        hypotheses = transition.next_hypotheses
        evidence = transition.next_evidence
        state = transition.next_controller_state
    raise AssertionError("recurrence fixture has no final round")


def _result_for_transition(
    hypotheses,
    state,
    proposal,
    actions,
    transition,
) -> ControllerResult:
    record = ControllerRoundRecord(
        hypotheses=hypotheses,
        controller_state=state,
        proposal=proposal,
        actions=actions,
        transition=transition,
        termination_output=None,
        termination=(TerminationDecision.CONTINUE,),
    )
    next_state = transition.next_controller_state
    return ControllerResult(
        hypotheses=transition.next_hypotheses,
        evidence=transition.next_evidence,
        termination=(TerminationDecision.UNKNOWN_INCOMPLETE,),
        selected_arc_trace=(),
        trace_ledger=next_state.trace_ledger,
        context_ledger=next_state.context_ledger,
        evidence_ledger=next_state.evidence_ledger,
        action_diagnostics=(),
        final_termination_logits=transition.next_evidence.new_zeros((1, 6)),
        rounds=next_state.round_index,
        arcs_scored=next_state.arcs_scored,
        contexts_read=next_state.contexts_read,
        round_records=(record,),
    )


def test_terminal_evidence_is_recorded_without_frontier_expansion() -> None:
    case, _ = generate_recurrence_necessity_pair(seed=715, horizon=4)
    batch = _batch(case)
    model = _model()
    controller = _controller()
    (
        oracle,
        hypotheses,
        evidence,
        state,
        proposal,
        supervision,
    ) = _final_recurrence_state(case, batch, model, controller)
    required = torch.nonzero(
        supervision.candidates.include_as_evidence,
        as_tuple=False,
    ).flatten()
    actions = ControllerActions(
        frontier_candidate_indices=torch.empty(
            0, dtype=torch.int64, device=batch.device
        ),
        context_candidate_indices=torch.empty(
            0, dtype=torch.int64, device=batch.device
        ),
        evidence_candidate_indices=required,
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
    report = observe_evidence_pipeline(
        batch,
        _result_for_transition(
            hypotheses,
            state,
            proposal,
            actions,
            transition,
        ),
        oracle,
    )

    assert transition.next_hypotheses.count == 0
    assert not transition.next_controller_state.trace_ledger[
        len(state.trace_ledger) :
    ]
    assert transition.next_controller_state.evidence_ledger[
        len(state.evidence_ledger) :
    ]
    required_rows = [
        row
        for row in report.requirement_observations
        if row.outstanding_before
    ]
    assert any(row.selected and row.recorded for row in required_rows)
    assert not any(row.frontier_selected for row in required_rows)
    assert report.exact_set_accuracy == 1.0


def test_edge_specific_metrics_reject_wrong_arc_to_correct_node() -> None:
    case, _ = generate_recurrence_necessity_pair(seed=716, horizon=4)
    batch = _batch(case)
    model = _model()
    controller = _controller()
    (
        oracle,
        hypotheses,
        evidence,
        state,
        proposal,
        supervision,
    ) = _final_recurrence_state(case, batch, model, controller)
    wrong = torch.nonzero(
        ~supervision.candidates.include_as_evidence,
        as_tuple=False,
    ).flatten()
    required_destination = case.evidence_nodes[0]
    node_offset = int(batch.graph.topology.graph_node_ptr[0].item())
    wrong = wrong[
        proposal.expansion.destination_node_ids[wrong].to(torch.int64)
        - node_offset
        == required_destination
    ][:1]
    assert wrong.numel() == 1
    actions = ControllerActions(
        frontier_candidate_indices=torch.empty(
            0, dtype=torch.int64, device=batch.device
        ),
        context_candidate_indices=torch.empty(
            0, dtype=torch.int64, device=batch.device
        ),
        evidence_candidate_indices=wrong,
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
    report = observe_evidence_pipeline(
        batch,
        _result_for_transition(
            hypotheses,
            state,
            proposal,
            actions,
            transition,
        ),
        oracle,
    )

    assert report.true_positives == 0
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.exact_set_accuracy == 0.0
