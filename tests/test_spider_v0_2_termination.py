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
    TerminationDecision,
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
)
from hippocampus.spider.terminator import TerminationOutput


def _fixture(*, learned_null: bool = False):
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    case = GraphProgramGenerator(
        GeneratorConfig(
            min_nodes=8,
            max_nodes=10,
            min_path_length=2,
            max_path_length=2,
        )
    ).generate(
        family=ProgramFamily.REACHABILITY,
        seed=9_011,
        answerable=True,
        require_multiple_paths=True,
    )
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=9_012)
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
            termination_mode="factorized",
            use_null_expansion=learned_null,
        )
    ).eval()
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=5,
            frontier_width=6,
            hypotheses_per_node=2,
            context_read_budget=4,
            evidence_selection_budget=4,
            search_budget=64,
            max_depth=6,
            expand_threshold=0.0,
            expansion_policy=(
                "learned_null" if learned_null else "threshold"
            ),
        )
    )
    return batch, model, controller


def _transition(batch, model, controller):
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    proposal = controller.propose(
        model,
        batch,
        hypotheses,
        evidence,
        state,
    )
    actions = ControllerActions(
        frontier_candidate_indices=torch.tensor(
            [0], dtype=torch.int64, device=batch.device
        ),
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
    return controller.apply(
        model,
        batch,
        hypotheses,
        evidence,
        state,
        proposal,
        actions,
    )


def _factorized_output(
    *,
    sufficient: float,
    useful: float,
    answer: float,
    unknown_index: int,
) -> TerminationOutput:
    unknown = torch.full((1, 4), -10.0)
    unknown[0, unknown_index] = 10.0
    return TerminationOutput(
        logits=torch.zeros((1, 6)),
        evidence_sufficient_logits=torch.tensor([sufficient]),
        useful_work_remaining_logits=torch.tensor([useful]),
        answer_supported_logits=torch.tensor([answer]),
        unknown_logits=unknown,
    )


def test_factorized_termination_cannot_stop_while_work_remains() -> None:
    batch, model, controller = _fixture()
    transition = _transition(batch, model, controller)

    premature_answer = _factorized_output(
        sufficient=-10.0,
        useful=10.0,
        answer=10.0,
        unknown_index=2,
    )
    supported_answer = _factorized_output(
        sufficient=10.0,
        useful=10.0,
        answer=10.0,
        unknown_index=2,
    )
    exhausted_conflict = _factorized_output(
        sufficient=-10.0,
        useful=-10.0,
        answer=-10.0,
        unknown_index=1,
    )

    assert controller.execute_termination(
        premature_answer,
        transition,
    ) == (TerminationDecision.CONTINUE,)
    assert controller.execute_termination(
        supported_answer,
        transition,
    ) == (TerminationDecision.ANSWER,)
    assert controller.execute_termination(
        exhausted_conflict,
        transition,
    ) == (TerminationDecision.UNKNOWN_CONFLICT,)


def test_learned_null_action_is_distinct_from_global_termination() -> None:
    batch, model, controller = _fixture(learned_null=True)
    hypotheses = model.initial_hypotheses(batch)
    state = ControllerState.initial()
    proposal = controller.propose(
        model,
        batch,
        hypotheses,
        model.initial_evidence(batch),
        state,
    )
    positive_expansion = replace(
        proposal.candidate_outputs,
        expand_logits=torch.full_like(
            proposal.candidate_outputs.expand_logits,
            10.0,
        ),
    )
    reject_all = replace(
        proposal,
        candidate_outputs=positive_expansion,
        null_expansion_logits=torch.tensor(
            [10.0], device=batch.device
        ),
    )
    allow_candidates = replace(
        reject_all,
        null_expansion_logits=torch.tensor(
            [-10.0], device=batch.device
        ),
    )

    null_actions = controller.choose_actions(
        reject_all,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(1),
    )
    candidate_actions = controller.choose_actions(
        allow_candidates,
        supervision=None,
        state=state,
        schedule=ActionSchedule.model_only(),
        randomizer=random.Random(1),
    )

    assert null_actions.frontier_candidate_indices.numel() == 0
    assert candidate_actions.frontier_candidate_indices.numel() > 0
    assert null_actions.termination_source is ActionSource.MODEL
