from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random

import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    RecurrenceNecessitySpec,
    build_recurrence_necessity_manifest,
    generate_recurrence_necessity_cases,
    generate_recurrence_necessity_pair,
    recurrence_metadata_leakage_report,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
    verify_case,
)
from hippocampus.spider import (
    ActionSchedule,
    ControllerState,
    SparseControllerConfig,
    SparseWavefrontController,
    SpiderModel,
    SpiderModelConfig,
    StateOracle,
)


def _out_degree(case, node_id: int) -> int:
    return sum(edge.source_node == node_id for edge in case.edges)


def test_recurrence_necessity_pair_is_deterministic_and_verified() -> None:
    first = generate_recurrence_necessity_pair(seed=711, horizon=6)
    repeated = generate_recurrence_necessity_pair(seed=711, horizon=6)

    assert first == repeated
    assert len(first) == 2
    for case in first:
        verify_case(case).raise_for_errors()
        assert len(case.trace.rounds) == 6
        assert all(
            round_.termination.decision.value == "continue"
            for round_ in case.trace.rounds[:-1]
        )
        assert case.trace.rounds[-1].termination.decision.value == "answer"
        assert len(case.evidence_edge_ids) == 1


def test_matched_pair_changes_history_binding_not_local_multisets() -> None:
    left, right = generate_recurrence_necessity_pair(seed=712, horizon=5)

    assert tuple(
        (edge.source_node, edge.destination_node)
        for edge in left.edges
    ) == tuple(
        (edge.source_node, edge.destination_node)
        for edge in right.edges
    )
    assert Counter(
        atom
        for edge in left.edges
        for atom in edge.atoms
    ) == Counter(
        atom
        for edge in right.edges
        for atom in edge.atoms
    )
    assert left.query_atoms == right.query_atoms
    assert left.nodes == right.nodes
    assert left.evidence_edge_ids != right.evidence_edge_ids


def test_first_hop_neighbours_have_matched_local_profiles() -> None:
    case, _ = generate_recurrence_necessity_pair(seed=713, horizon=8)
    first_round = case.trace.rounds[0]
    neighbours = [
        candidate.destination_node for candidate in first_round.candidates
    ]

    assert len(neighbours) >= 3
    assert len({case.nodes[node].summary_atoms for node in neighbours}) == 1
    assert len({_out_degree(case, node) for node in neighbours}) == 1
    assert len(
        {
            len(case.nodes[node].summary_atoms)
            for node in neighbours
        }
    ) == 1
    assert len(
        {
            len(case.edges[candidate.edge_id].atoms)
            for candidate in first_round.candidates
        }
    ) == 1


def test_all_branches_stay_acceptable_until_final_evidence_comparison() -> None:
    case, _ = generate_recurrence_necessity_pair(seed=714, horizon=7)

    for round_ in case.trace.rounds:
        assert all(candidate.acceptable for candidate in round_.candidates)
    final = case.trace.rounds[-1]
    assert sum(
        candidate.include_as_evidence for candidate in final.candidates
    ) == 1
    assert {
        candidate.destination_node for candidate in final.candidates
    } == set(case.evidence_nodes)


def test_recurrence_split_manifest_is_disjoint_and_stable() -> None:
    train = RecurrenceNecessitySpec(
        name="train_recurrence_necessity",
        case_count=32,
        seed_start=810_000,
    )
    validation = RecurrenceNecessitySpec(
        name="validation_recurrence_necessity",
        case_count=32,
        seed_start=820_000,
    )
    train_manifest = build_recurrence_necessity_manifest(train)
    repeated = build_recurrence_necessity_manifest(train)
    validation_manifest = build_recurrence_necessity_manifest(validation)

    assert train_manifest == repeated
    assert train_manifest.sha256 != validation_manifest.sha256
    assert not (
        set(train_manifest.case_ids) & set(validation_manifest.case_ids)
    )
    assert len(generate_recurrence_necessity_cases(train)) == 32


def test_metadata_only_final_position_baseline_remains_near_chance() -> None:
    spec = RecurrenceNecessitySpec(
        name="development_recurrence_leakage",
        case_count=128,
        seed_start=830_000,
    )
    cases = generate_recurrence_necessity_cases(spec)
    report = recurrence_metadata_leakage_report(cases)

    assert report.case_count == 128
    assert report.chance_accuracy == 1 / spec.branch_count
    assert report.final_position_accuracy < 0.45
    assert report.first_hop_profile_mismatch_count == 0


def test_state_oracle_does_not_credit_wrong_arc_to_right_answer_node() -> None:
    case, _ = generate_recurrence_necessity_pair(seed=715, horizon=4)
    schema = GraphSchema(summary_dim=8, context_dim=8, edge_dim=8)
    renderer = SyntheticManifoldRenderer(schema, query_dim=8, seed=92)
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
        )
    ).eval()
    controller = SparseWavefrontController(
        SparseControllerConfig(
            max_rounds=8,
            frontier_width=8,
            hypotheses_per_node=2,
            context_read_budget=0,
            evidence_selection_budget=8,
            search_budget=256,
            max_depth=10,
        )
    )
    oracle = StateOracle(case, batch, controller.config)
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    final_proposal = None
    final_supervision = None
    for _ in range(4):
        proposal = controller.propose(
            model,
            batch,
            hypotheses,
            evidence,
            state,
        )
        supervision = oracle.label(proposal, hypotheses, state)
        actions = controller.choose_actions(
            proposal,
            supervision=supervision,
            state=state,
            schedule=ActionSchedule.oracle_only(),
            randomizer=random.Random(0),
        )
        if state.round_index == 3:
            final_proposal = proposal
            final_supervision = supervision
            final_hypotheses = hypotheses
            final_evidence = evidence
            final_state = state
            final_actions = actions
            break
        transition = controller.apply(
            model,
            batch,
            hypotheses,
            evidence,
            state,
            proposal,
            actions,
        )
        hypotheses = transition.next_hypotheses
        evidence = transition.next_evidence
        state = transition.next_controller_state

    assert final_proposal is not None
    assert final_supervision is not None
    correct = torch.nonzero(
        final_supervision.candidates.include_as_evidence,
        as_tuple=False,
    ).flatten()
    wrong = torch.nonzero(
        ~final_supervision.candidates.include_as_evidence,
        as_tuple=False,
    ).flatten()[:1]
    wrong_transition = controller.apply(
        model,
        batch,
        final_hypotheses,
        final_evidence,
        final_state,
        final_proposal,
        replace(final_actions, evidence_candidate_indices=wrong),
    )
    correct_transition = controller.apply(
        model,
        batch,
        final_hypotheses,
        final_evidence,
        final_state,
        final_proposal,
        replace(final_actions, evidence_candidate_indices=correct),
    )

    assert (
        oracle.termination_target(wrong_transition).decision.value
        != "answer"
    )
    assert (
        oracle.termination_target(correct_transition).decision.value
        == "answer"
    )
