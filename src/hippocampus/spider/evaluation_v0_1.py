from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import random
import time
from typing import Sequence

import torch

from ..programs.batching import PackedProgramBatch
from ..programs.schema import ProgramFamily, TerminationDecision
from ..programs.stress import RolloutStressExample, RolloutStressKind
from .calibration import (
    EvidenceCalibration,
    binary_average_precision,
    calibrate_evidence_threshold,
    precision_recall_curve,
    validate_calibration_source,
)
from .config import SparseControllerConfig
from .controller import (
    ActionSchedule,
    ContextLedgerEntry,
    ControllerState,
    EvidenceLedgerEntry,
    SparseWavefrontController,
)
from .evidence_diagnostics import (
    EvidencePipelineCaseReport,
    aggregate_evidence_pipeline,
    observe_evidence_pipeline,
)
from .execution import ControllerExecutionPolicy
from .hypothesis import HypothesisBatch
from .model import CandidateScorerBase
from .state_oracle import StateOracle
from .training import evaluate_oracle_batches


_DECISIONS = tuple(TerminationDecision)
_DECISION_INDEX = {
    decision: index for index, decision in enumerate(_DECISIONS)
}


@dataclass(frozen=True, slots=True)
class RoundTerminationObservation:
    round_index: int
    target: TerminationDecision
    prediction: TerminationDecision


@dataclass(frozen=True, slots=True)
class ClosedLoopCaseExecution:
    prediction: TerminationDecision
    final_logits: torch.Tensor
    trace_ledger: tuple[object, ...]
    context_ledger: tuple[object, ...]
    evidence_ledger: tuple[object, ...]
    rounds: int
    arcs_scored: int
    contexts_read: int
    candidate_evidence_logits: torch.Tensor
    candidate_evidence_labels: torch.Tensor
    scored_positive_nodes: tuple[int, ...]
    frontier_selected_nodes: tuple[int, ...]
    round_termination: tuple[RoundTerminationObservation, ...]
    evidence_pipeline: EvidencePipelineCaseReport


@dataclass(frozen=True, slots=True)
class ClosedLoopEvaluationReport:
    split: str
    dataset_version: str
    case_count: int
    evidence_threshold: float
    primary_autonomous_success: float
    fixed_horizon_structural_success: float
    execution: dict[str, object]
    teacher_forced: dict[str, float | int]
    rollout: dict[str, object]
    evidence: dict[str, object]
    evidence_pipeline: dict[str, object]
    termination: dict[str, object]
    efficiency: dict[str, float | int]
    invariance: dict[str, float | int]
    per_family: dict[str, dict[str, float | int]]
    runtime_seconds: float
    peak_cuda_memory_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def execute_closed_loop_case(
    model: CandidateScorerBase,
    batch: PackedProgramBatch,
    controller_config: SparseControllerConfig,
    execution_policy: ControllerExecutionPolicy | None = None,
) -> ClosedLoopCaseExecution:
    if batch.graph_count != 1:
        raise ValueError("closed-loop evaluation requires singleton batches")
    controller = SparseWavefrontController(controller_config)
    policy = execution_policy or ControllerExecutionPolicy.learned()
    oracle = StateOracle(batch.cases[0], batch, controller_config)
    result = controller.run(
        model,
        batch,
        execution_policy=policy,
    )
    evidence_logits: list[torch.Tensor] = []
    evidence_labels: list[torch.Tensor] = []
    scored_positive_nodes: set[int] = set()
    frontier_selected_nodes: set[int] = set()
    round_observations: list[RoundTerminationObservation] = []
    node_offset = int(batch.graph.topology.graph_node_ptr[0].item())
    for record in result.round_records:
        supervision = oracle.label(
            record.proposal,
            record.hypotheses,
            record.controller_state,
        )
        termination_target = oracle.termination_target(
            record.transition
        ).decision
        if record.termination_output is not None:
            round_observations.append(
                RoundTerminationObservation(
                    round_index=record.controller_state.round_index,
                    target=termination_target,
                    prediction=record.termination[0],
                )
            )
        evidence_logits.append(
            record.transition.refined_outputs.evidence_logits.detach()
        )
        evidence_labels.append(
            supervision.candidates.include_as_evidence.detach()
        )
        destinations = (
            record.proposal.expansion.destination_node_ids - node_offset
        )
        positive_destinations = destinations[
            supervision.candidates.include_as_evidence
        ]
        scored_positive_nodes.update(positive_destinations.tolist())
        if record.actions.frontier_candidate_indices.numel():
            frontier_selected_nodes.update(
                destinations[
                    record.actions.frontier_candidate_indices
                ].tolist()
            )

    return ClosedLoopCaseExecution(
        prediction=result.termination[0],
        final_logits=result.final_termination_logits,
        trace_ledger=result.trace_ledger,
        context_ledger=result.context_ledger,
        evidence_ledger=result.evidence_ledger,
        rounds=result.rounds,
        arcs_scored=result.arcs_scored,
        contexts_read=result.contexts_read,
        candidate_evidence_logits=(
            torch.cat(evidence_logits)
            if evidence_logits
            else result.evidence.new_empty((0,))
        ),
        candidate_evidence_labels=(
            torch.cat(evidence_labels)
            if evidence_labels
            else torch.empty(0, dtype=torch.bool, device=batch.device)
        ),
        scored_positive_nodes=tuple(sorted(scored_positive_nodes)),
        frontier_selected_nodes=tuple(sorted(frontier_selected_nodes)),
        round_termination=tuple(round_observations),
        evidence_pipeline=observe_evidence_pipeline(
            batch,
            result,
            oracle,
        ),
    )


def calibrate_on_development_batches(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    controller_config: SparseControllerConfig,
    split_name: str = "validation_id",
    dataset_version: str = "spider-programs-v0.2",
    execution_policy: ControllerExecutionPolicy | None = None,
) -> EvidenceCalibration:
    validate_calibration_source(
        split_name=split_name,
        dataset_version=dataset_version,
    )
    if not batches:
        raise ValueError("calibration requires development batches")
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            execution = execute_closed_loop_case(
                model,
                batch,
                controller_config,
                execution_policy,
            )
            scores.append(execution.candidate_evidence_logits)
            labels.append(execution.candidate_evidence_labels)
    model.train(was_training)
    return calibrate_evidence_threshold(
        torch.cat(scores),
        torch.cat(labels),
        split_name=split_name,
        dataset_version=dataset_version,
    )


def _stress_hypotheses(
    model: CandidateScorerBase,
    batch: PackedProgramBatch,
    example: RolloutStressExample,
) -> HypothesisBatch:
    template = model.initial_hypotheses(batch)
    count = len(example.frontier_nodes)
    device = batch.device
    node_offset = int(batch.graph.topology.graph_node_ptr[0].item())
    node_ids = torch.tensor(
        [node_offset + node for node in example.frontier_nodes],
        dtype=torch.int32,
        device=device,
    )
    path = template.path_state[:1].expand(count, -1, -1)
    return HypothesisBatch(
        node_ids=node_ids,
        graph_ids=torch.zeros(count, dtype=torch.int32, device=device),
        path_state=path,
        scores=path.new_zeros((count,)),
        depths=torch.full(
            (count,),
            example.round_index,
            dtype=torch.int32,
            device=device,
        ),
        parent_trace_ids=torch.full(
            (count,),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        incoming_arc_ids=torch.full(
            (count,),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        incoming_edge_ids=torch.full(
            (count,),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        context_read=torch.zeros(count, dtype=torch.bool, device=device),
    ).validate()


def evaluate_rollout_stress_states(
    model: CandidateScorerBase,
    examples: Sequence[RolloutStressExample],
    batches: Sequence[PackedProgramBatch],
    *,
    controller_config: SparseControllerConfig,
) -> dict[str, object]:
    """Execute the shared transition from supervisor-defined dev states."""

    if len(examples) != len(batches) or not examples:
        raise ValueError("stress examples and singleton batches must align")
    per_kind = {
        kind.value: {
            "count": 0,
            "recoverability_correct": 0,
            "termination_correct": 0,
        }
        for kind in RolloutStressKind
    }
    recoverability_correct = 0
    termination_correct = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for example, batch in zip(examples, batches, strict=True):
            controller = SparseWavefrontController(controller_config)
            oracle = StateOracle(example.case, batch, controller_config)
            hypotheses = _stress_hypotheses(model, batch, example)
            evidence = model.initial_evidence(batch)
            node_offset = int(
                batch.graph.topology.graph_node_ptr[0].item()
            )
            context_ledger = tuple(
                ContextLedgerEntry(
                    node_id=node_offset + node,
                    edge_id=-1,
                    arc_id=-1,
                    round_index=max(0, example.round_index - 1),
                    frontier_position=-1,
                    parent_trace_id=-1,
                )
                for node in example.contexts_read_nodes
            )
            evidence_ledger = tuple(
                EvidenceLedgerEntry(
                    node_id=node_offset + node,
                    edge_id=-1,
                    arc_id=-1,
                    round_index=max(0, example.round_index - 1),
                    frontier_position=-1,
                    parent_trace_id=-1,
                    context_read=node in example.contexts_read_nodes,
                )
                for node in example.accumulated_evidence_nodes
            )
            search_limit, context_limit, evidence_limit = (
                controller.resolved_limits(batch)
            )
            state = ControllerState(
                round_index=example.round_index,
                arcs_scored=example.arcs_scored,
                contexts_read=example.contexts_read,
                evidence_selected=len(evidence_ledger),
                search_budget_exhausted=(
                    example.arcs_scored >= search_limit
                ),
                context_budget_exhausted=(
                    example.contexts_read >= context_limit
                ),
                evidence_budget_exhausted=(
                    len(evidence_ledger) >= evidence_limit
                ),
                frontier_empty=hypotheses.count == 0,
                last_expansion_had_arcs=False,
                last_expansion_truncated=False,
                deliberate_empty_frontier=False,
                depth_exhausted=False,
                trace_ledger=(),
                context_ledger=context_ledger,
                evidence_ledger=evidence_ledger,
            )
            proposal = controller.propose(
                model,
                batch,
                hypotheses,
                evidence,
                state,
            )
            supervision = oracle.label(proposal, hypotheses, state)
            selection = controller.select_actions(
                model,
                batch,
                proposal,
                supervision=None,
                state=state,
                schedule=ActionSchedule.model_only(),
                randomizer=random.Random(0),
            )
            proposal = selection.proposal
            actions = selection.actions
            transition = controller.apply(
                model,
                batch,
                hypotheses,
                evidence,
                state,
                proposal,
                actions,
            )
            target = oracle.termination_target(transition).decision
            output = model.termination_output(
                batch,
                transition.next_hypotheses,
                transition.next_evidence,
                transition.termination_control,
            )
            logits = output.logits
            prediction = controller.execute_termination(
                output,
                transition,
            )[0]
            recoverable_match = (
                supervision.recoverable == example.expected_recoverable
            )
            termination_match = prediction is target
            recoverability_correct += int(recoverable_match)
            termination_correct += int(termination_match)
            metrics = per_kind[example.kind.value]
            metrics["count"] += 1
            metrics["recoverability_correct"] += int(recoverable_match)
            metrics["termination_correct"] += int(termination_match)
    model.train(was_training)
    for metrics in per_kind.values():
        count = max(1, int(metrics["count"]))
        metrics["recoverability_accuracy"] = (
            int(metrics["recoverability_correct"]) / count
        )
        metrics["termination_accuracy"] = (
            int(metrics["termination_correct"]) / count
        )
    return {
        "case_count": len(examples),
        "recoverability_accuracy": recoverability_correct / len(examples),
        "termination_accuracy": termination_correct / len(examples),
        "per_kind": per_kind,
    }


def _valid_wavefront(
    batch: PackedProgramBatch,
    trace_ledger: tuple[object, ...],
) -> bool:
    case = batch.cases[0]
    if not case.answerable:
        return True
    edge_offset = int(batch.graph.topology.graph_edge_ptr[0].item())
    selected_by_round: dict[int, set[int]] = {}
    for entry in trace_ledger:
        selected_by_round.setdefault(entry.round_index, set()).add(
            entry.edge_id - edge_offset
        )
    for round_index, round_ in enumerate(case.trace.rounds):
        acceptable = {
            candidate.edge_id
            for candidate in round_.candidates
            if candidate.acceptable
        }
        if acceptable and not (
            acceptable & selected_by_round.get(round_index, set())
        ):
            return False
    return True


def _confusion() -> list[list[int]]:
    return [
        [0 for _ in _DECISIONS]
        for _ in _DECISIONS
    ]


def _add_confusion(
    matrix: list[list[int]],
    target: TerminationDecision,
    prediction: TerminationDecision,
) -> None:
    matrix[_DECISION_INDEX[target]][_DECISION_INDEX[prediction]] += 1


def _risk_at_coverage(
    cases: list[tuple[float, bool]],
    coverage: float,
) -> float:
    if not cases:
        return 0.0
    count = max(1, math.ceil(len(cases) * coverage))
    selected = sorted(cases, key=lambda item: item[0], reverse=True)[:count]
    return sum(not correct for _, correct in selected) / count


def _invariance(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    permuted: Sequence[PackedProgramBatch] | None,
    controller_config: SparseControllerConfig,
    *,
    sample_limit: int,
    execution_policy: ControllerExecutionPolicy | None = None,
) -> dict[str, float | int]:
    controller = SparseWavefrontController(controller_config)
    samples = min(sample_limit, len(batches))
    replay_mismatches = 0
    row_mismatches = 0
    max_delta = 0.0
    with torch.no_grad():
        for index in range(samples):
            first = controller.run(
                model,
                batches[index],
                execution_policy=execution_policy,
            )
            repeated = controller.run(
                model,
                batches[index],
                execution_policy=execution_policy,
            )
            replay_mismatches += int(
                first.selected_arc_trace != repeated.selected_arc_trace
                or first.termination != repeated.termination
                or not torch.equal(
                    first.final_termination_logits,
                    repeated.final_termination_logits,
                )
            )
            if permuted is None:
                continue
            other = controller.run(
                model,
                permuted[index],
                execution_policy=execution_policy,
            )
            if first.final_termination_logits.numel():
                max_delta = max(
                    max_delta,
                    float(
                        (
                            first.final_termination_logits.float()
                            - other.final_termination_logits.float()
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                )
            row_mismatches += int(
                first.selected_arc_trace != other.selected_arc_trace
                or first.termination != other.termination
            )
    return {
        "deterministic_replay_samples": samples,
        "deterministic_replay_mismatches": replay_mismatches,
        "row_permutation_samples": samples if permuted is not None else 0,
        "row_permutation_decision_mismatches": row_mismatches,
        "maximum_score_delta": max_delta,
    }


def evaluate_closed_loop_batches(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    split: str,
    controller_config: SparseControllerConfig,
    dataset_version: str = "spider-programs-v0.2",
    evidence_threshold: float | None = None,
    permuted_batches: Sequence[PackedProgramBatch] | None = None,
    invariance_sample_limit: int = 8,
    execution_policy: ControllerExecutionPolicy | None = None,
    include_teacher_forced: bool = True,
) -> ClosedLoopEvaluationReport:
    if not batches:
        raise ValueError("evaluation requires at least one batch")
    if permuted_batches is not None and len(permuted_batches) != len(batches):
        raise ValueError("permuted batches must align with evaluation batches")
    threshold = (
        controller_config.evidence_threshold
        if evidence_threshold is None
        else evidence_threshold
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("evidence threshold must be in [0, 1]")
    resolved_controller = replace(
        controller_config,
        evidence_threshold=threshold,
    )
    policy = execution_policy or ControllerExecutionPolicy.learned()
    was_training = model.training
    model.eval()
    device = batches[0].device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if include_teacher_forced:
        oracle_loss, teacher = evaluate_oracle_batches(
            model,
            batches,
            controller_config=resolved_controller,
        )
        teacher_forced: dict[str, float | int] = {
            "oracle_loss": oracle_loss,
            **teacher.as_dict(),
        }
    else:
        teacher_forced = {
            "skipped": 1,
        }

    termination_correct = 0
    unknown_correct = 0
    unknown_count = 0
    answered = 0
    answered_errors = 0
    false_answers = 0
    false_unknowns = 0
    autonomous_success = 0
    structural_success = 0
    valid_paths = 0
    answerable_count = 0
    trace_valid = 0
    semantic_invalid = 0
    selected_arcs = 0
    evidence_tp = 0
    evidence_fp = 0
    evidence_fn = 0
    evidence_exact = 0
    context_tp = 0
    context_fp = 0
    context_fn = 0
    arcs_scored = 0
    contexts_read = 0
    rounds = 0
    one_round_stops = 0
    latency: list[float] = []
    all_evidence_scores: list[torch.Tensor] = []
    all_evidence_labels: list[torch.Tensor] = []
    confidence_correct: list[tuple[float, bool]] = []
    overall_confusion = _confusion()
    round_confusions: dict[int, list[list[int]]] = {}
    family_confusions = {
        family.value: _confusion() for family in ProgramFamily
    }
    family_counts = {family.value: 0 for family in ProgramFamily}
    family_success = {family.value: 0 for family in ProgramFamily}
    family_evidence_tp = {family.value: 0 for family in ProgramFamily}
    family_evidence_total = {family.value: 0 for family in ProgramFamily}
    pipeline_reports: list[EvidencePipelineCaseReport] = []

    with torch.no_grad():
        for batch in batches:
            case_started = time.perf_counter()
            execution = execute_closed_loop_case(
                model,
                batch,
                resolved_controller,
                policy,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency.append(time.perf_counter() - case_started)
            case = batch.cases[0]
            expected = case.termination.decision
            prediction = execution.prediction
            correct = prediction is expected
            termination_correct += int(correct)
            _add_confusion(overall_confusion, expected, prediction)
            _add_confusion(
                family_confusions[case.family.value],
                expected,
                prediction,
            )
            family_counts[case.family.value] += 1
            if expected is not TerminationDecision.ANSWER:
                unknown_count += 1
                unknown_correct += int(correct)
            if prediction is TerminationDecision.ANSWER:
                answered += 1
                answered_errors += int(not correct)
                false_answers += int(expected is not TerminationDecision.ANSWER)
            elif expected is TerminationDecision.ANSWER:
                false_unknowns += 1

            probabilities = torch.softmax(
                execution.final_logits.float(),
                dim=-1,
            )
            confidence_correct.append(
                (float(probabilities.max().item()), correct)
            )
            node_offset = int(
                batch.graph.topology.graph_node_ptr[0].item()
            )
            edge_offset = int(
                batch.graph.topology.graph_edge_ptr[0].item()
            )
            pipeline = execution.evidence_pipeline
            pipeline_reports.append(pipeline)
            exact_evidence_match = bool(pipeline.exact_set_accuracy)
            tp = pipeline.true_positives
            fp = pipeline.false_positives
            fn = pipeline.false_negatives
            evidence_tp += tp
            evidence_fp += fp
            evidence_fn += fn
            evidence_exact += int(exact_evidence_match)
            family_evidence_tp[case.family.value] += tp
            family_evidence_total[case.family.value] += (
                pipeline.required_cardinality
            )

            predicted_context = {
                entry.node_id - node_offset
                for entry in execution.context_ledger
            }
            expected_context = {
                candidate.destination_node
                for round_ in case.trace.rounds
                for candidate in round_.candidates
                if candidate.context_has_value
            }
            context_tp += len(predicted_context & expected_context)
            context_fp += len(predicted_context - expected_context)
            context_fn += len(expected_context - predicted_context)

            valid_path = _valid_wavefront(batch, execution.trace_ledger)
            if case.answerable:
                answerable_count += 1
                valid_paths += int(valid_path)
            trace_is_valid = True
            for entry in execution.trace_ledger:
                selected_arcs += 1
                local_edge = entry.edge_id - edge_offset
                if not 0 <= local_edge < len(case.edges):
                    trace_is_valid = False
                    continue
                semantic_invalid += int(not case.edges[local_edge].valid)
            trace_valid += int(trace_is_valid)
            structural_success += int(
                trace_is_valid
                and exact_evidence_match
                and (valid_path if case.answerable else True)
            )

            case_success = (
                prediction is TerminationDecision.ANSWER
                and valid_path
                and pipeline.false_negatives == 0
                if case.answerable
                else prediction is expected
                and prediction is not TerminationDecision.ANSWER
            )
            autonomous_success += int(case_success)
            family_success[case.family.value] += int(case_success)
            arcs_scored += execution.arcs_scored
            contexts_read += execution.contexts_read
            rounds += execution.rounds
            one_round_stops += int(execution.rounds == 1)
            all_evidence_scores.append(
                execution.candidate_evidence_logits
            )
            all_evidence_labels.append(
                execution.candidate_evidence_labels
            )
            for observation in execution.round_termination:
                matrix = round_confusions.setdefault(
                    observation.round_index,
                    _confusion(),
                )
                _add_confusion(
                    matrix,
                    observation.target,
                    observation.prediction,
                )

    evidence_scores = torch.cat(all_evidence_scores)
    evidence_labels = torch.cat(all_evidence_labels)
    pipeline_report = aggregate_evidence_pipeline(pipeline_reports)
    pipeline_overall = pipeline_report["overall"]
    evidence_precision = evidence_tp / max(1, evidence_tp + evidence_fp)
    evidence_recall = evidence_tp / max(1, evidence_tp + evidence_fn)
    evidence_f1 = (
        2 * evidence_precision * evidence_recall
        / max(1e-12, evidence_precision + evidence_recall)
    )
    curve = precision_recall_curve(evidence_scores, evidence_labels)
    evidence_report: dict[str, object] = {
        "positive_label_count": int(evidence_labels.sum().item()),
        "negative_label_count": int((~evidence_labels).sum().item()),
        "precision": evidence_precision,
        "recall": evidence_recall,
        "f1": evidence_f1,
        "average_precision": binary_average_precision(
            evidence_scores,
            evidence_labels,
        ),
        "exact_set_accuracy": evidence_exact / len(batches),
        "recall_conditioned_on_scored": (
            pipeline_overall["selection_recall_conditioned_on_scored"]
        ),
        "recall_conditioned_on_frontier_selection": (
            pipeline_overall["frontier_selection_recall"]
        ),
        "operating_threshold": threshold,
        "precision_recall_curve": [
            asdict(point) for point in curve
        ],
    }
    coverage = answered / len(batches)
    rollout: dict[str, object] = {
        "termination_accuracy": termination_correct / len(batches),
        "unknown_reason_accuracy": unknown_correct / max(1, unknown_count),
        "answered_coverage": coverage,
        "risk_among_answered": answered_errors / max(1, answered),
        "risk_at_25_percent_coverage": _risk_at_coverage(
            confidence_correct,
            0.25,
        ),
        "risk_at_50_percent_coverage": _risk_at_coverage(
            confidence_correct,
            0.50,
        ),
        "risk_at_75_percent_coverage": _risk_at_coverage(
            confidence_correct,
            0.75,
        ),
        "false_answer_rate": false_answers / len(batches),
        "false_unknown_rate": false_unknowns / len(batches),
        "exact_valid_path_rate": valid_paths / max(1, answerable_count),
        "trace_validity": trace_valid / len(batches),
        "semantic_invalid_expansion_rate": (
            semantic_invalid / max(1, selected_arcs)
        ),
        "one_round_stop_rate": one_round_stops / len(batches),
    }
    termination_report: dict[str, object] = {
        "labels": [decision.value for decision in _DECISIONS],
        "overall_confusion": overall_confusion,
        "per_round_confusion": {
            str(round_index): matrix
            for round_index, matrix in sorted(round_confusions.items())
        },
        "per_family_confusion": family_confusions,
        "continue_stop_accuracy": sum(
            int(
                (target_index == 0) == (prediction_index == 0)
            )
            * count
            for target_index, row in enumerate(overall_confusion)
            for prediction_index, count in enumerate(row)
        )
        / len(batches),
        "answer_unknown_accuracy_on_stop_targets": (
            sum(
                int(
                    (target_index == 1)
                    == (prediction_index == 1)
                )
                * count
                for target_index, row in enumerate(overall_confusion)
                if target_index >= 1
                for prediction_index, count in enumerate(row)
            )
            / max(
                1,
                sum(
                    count
                    for target_index, row in enumerate(overall_confusion)
                    if target_index >= 1
                    for count in row
                ),
            )
        ),
        "unknown_reason_accuracy_on_unknown_targets": (
            sum(
                overall_confusion[index][index]
                for index in range(2, len(_DECISIONS))
            )
            / max(
                1,
                sum(
                    count
                    for target_index, row in enumerate(overall_confusion)
                    if target_index >= 2
                    for count in row
                ),
            )
        ),
        "per_class_recall": {
            decision.value: (
                overall_confusion[index][index]
                / max(1, sum(overall_confusion[index]))
            )
            for index, decision in enumerate(_DECISIONS)
        },
    }
    per_family = {
        family.value: {
            "case_count": family_counts[family.value],
            "autonomous_success": (
                family_success[family.value]
                / max(1, family_counts[family.value])
            ),
            "evidence_recall": (
                family_evidence_tp[family.value]
                / max(1, family_evidence_total[family.value])
            ),
        }
        for family in ProgramFamily
    }
    invariance = _invariance(
        model,
        batches,
        permuted_batches,
        resolved_controller,
        sample_limit=invariance_sample_limit,
        execution_policy=policy,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory = 0
    runtime = time.perf_counter() - started
    model.train(was_training)
    return ClosedLoopEvaluationReport(
        split=split,
        dataset_version=dataset_version,
        case_count=len(batches),
        evidence_threshold=threshold,
        primary_autonomous_success=autonomous_success / len(batches),
        fixed_horizon_structural_success=(
            structural_success / len(batches)
        ),
        execution={
            "horizon_mode": policy.horizon_mode.value,
            "fixed_rounds": policy.fixed_rounds,
            "path_state_intervention": (
                policy.path_state_intervention.value
            ),
            "intervention_seed": policy.intervention_seed,
            "intermediate_termination_suppressed": (
                policy.suppresses_intermediate_termination
            ),
        },
        teacher_forced=teacher_forced,
        rollout=rollout,
        evidence=evidence_report,
        evidence_pipeline=pipeline_report,
        termination=termination_report,
        efficiency={
            "mean_rounds": rounds / len(batches),
            "mean_arcs_scored": arcs_scored / len(batches),
            "mean_contexts_read": contexts_read / len(batches),
            "mean_wall_clock_latency_seconds": sum(latency) / len(latency),
            "peak_cuda_memory_bytes": peak_memory,
        },
        invariance=invariance,
        per_family=per_family,
        runtime_seconds=runtime,
        peak_cuda_memory_bytes=peak_memory,
    )
