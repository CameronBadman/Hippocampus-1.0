from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch

from ..programs.batching import PackedProgramBatch
from ..programs.schema import TerminationDecision
from .config import SparseControllerConfig
from .controller import SparseWavefrontController
from .model import CandidateScorerBase
from .training import OracleMetrics, evaluate_oracle_batches


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    split: str
    case_count: int
    oracle_loss: float
    teacher_forced: dict[str, float | int]
    rollout: dict[str, float | int]
    efficiency: dict[str, float | int]
    invariance: dict[str, float | int]
    runtime_seconds: float
    peak_cuda_memory_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "case_count": self.case_count,
            "oracle_loss": self.oracle_loss,
            "teacher_forced": self.teacher_forced,
            "rollout": self.rollout,
            "efficiency": self.efficiency,
            "invariance": self.invariance,
            "runtime_seconds": self.runtime_seconds,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
        }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / max(1.0, denominator)


def _expected_context_nodes(batch: PackedProgramBatch) -> set[int]:
    return {
        candidate.destination_node
        for round_ in batch.cases[0].trace.rounds
        for candidate in round_.candidates
        if candidate.context_has_value
    }


def _valid_wavefront(
    batch: PackedProgramBatch,
    selected_by_round: dict[int, set[int]],
) -> bool:
    case = batch.cases[0]
    if not case.answerable:
        return True
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


def _invariance_metrics(
    model: CandidateScorerBase,
    original: Sequence[PackedProgramBatch],
    permuted: Sequence[PackedProgramBatch] | None,
    controller: SparseWavefrontController,
    *,
    sample_limit: int,
) -> dict[str, float | int]:
    max_score_delta = 0.0
    decision_mismatches = 0
    replay_mismatches = 0
    samples = min(sample_limit, len(original))
    with torch.no_grad():
        for index in range(samples):
            batch = original[index]
            first_run = controller.run(model, batch)
            second_run = controller.run(model, batch)
            if (
                first_run.selected_arc_trace != second_run.selected_arc_trace
                or first_run.termination != second_run.termination
            ):
                replay_mismatches += 1
            if permuted is None:
                continue
            other = permuted[index]

            def candidate_scores(candidate_batch: PackedProgramBatch):
                hypotheses = model.initial_hypotheses(candidate_batch)
                expansion = candidate_batch.graph.expand_frontier(
                    hypotheses.node_ids
                )
                outputs = model.score_candidates(
                    candidate_batch,
                    hypotheses,
                    expansion,
                    model.initial_evidence(candidate_batch),
                )
                return expansion, outputs

            first_expansion, first = candidate_scores(batch)
            second_expansion, second = candidate_scores(other)
            if not torch.equal(first_expansion.arc_ids, second_expansion.arc_ids):
                decision_mismatches += 1
                continue
            if first.candidate_count:
                delta = (
                    first.priority_logits.float()
                    - second.priority_logits.float()
                ).abs().max()
                max_score_delta = max(max_score_delta, float(delta.item()))
                first_decision = (
                    int(first.priority_logits.argmax().item()),
                    tuple((first.expand_logits >= 0).tolist()),
                )
                second_decision = (
                    int(second.priority_logits.argmax().item()),
                    tuple((second.expand_logits >= 0).tolist()),
                )
                decision_mismatches += int(first_decision != second_decision)
    return {
        "row_permutation_samples": samples if permuted is not None else 0,
        "maximum_score_delta": max_score_delta,
        "row_permutation_decision_mismatches": decision_mismatches,
        "deterministic_replay_samples": samples,
        "deterministic_replay_mismatches": replay_mismatches,
    }


def evaluate_batches(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    split: str,
    controller_config: SparseControllerConfig,
    permuted_batches: Sequence[PackedProgramBatch] | None = None,
    invariance_sample_limit: int = 8,
) -> EvaluationReport:
    if not batches:
        raise ValueError("evaluation requires at least one packed batch")
    if permuted_batches is not None and len(permuted_batches) != len(batches):
        raise ValueError("permuted batches must align with original batches")
    was_training = model.training
    model.eval()
    if batches[0].device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(batches[0].device)
        torch.cuda.synchronize(batches[0].device)
    started = time.perf_counter()
    oracle_loss, teacher = evaluate_oracle_batches(
        model,
        batches,
        controller_config=controller_config,
    )
    controller = SparseWavefrontController(controller_config)

    termination_correct = 0
    unknown_reason_correct = 0
    unknown_count = 0
    answered = 0
    answered_errors = 0
    false_answers = 0
    false_unknowns = 0
    evidence_tp = 0
    evidence_fp = 0
    evidence_fn = 0
    exact_evidence = 0
    context_tp = 0
    context_fp = 0
    context_fn = 0
    valid_paths = 0
    answerable_count = 0
    semantic_invalid = 0
    selected_count = 0
    trace_valid_count = 0
    arcs_scored = 0
    contexts_read = 0
    rounds = 0
    nodes_expanded = 0
    per_case_latency: list[float] = []

    with torch.no_grad():
        for batch in batches:
            case_started = time.perf_counter()
            result = controller.run(model, batch)
            if batch.device.type == "cuda":
                torch.cuda.synchronize(batch.device)
            per_case_latency.append(time.perf_counter() - case_started)
            case = batch.cases[0]
            prediction = result.termination[0]
            expected = case.termination.decision
            termination_correct += int(prediction is expected)
            if expected is not TerminationDecision.ANSWER:
                unknown_count += 1
                unknown_reason_correct += int(prediction is expected)
            if prediction is TerminationDecision.ANSWER:
                answered += 1
                error = expected is not TerminationDecision.ANSWER
                answered_errors += int(error)
                false_answers += int(error)
            elif expected is TerminationDecision.ANSWER:
                false_unknowns += 1

            predicted_evidence = {
                entry.node_id for entry in result.evidence_ledger
            }
            expected_evidence = set(case.evidence_nodes)
            evidence_tp += len(predicted_evidence & expected_evidence)
            evidence_fp += len(predicted_evidence - expected_evidence)
            evidence_fn += len(expected_evidence - predicted_evidence)
            exact_evidence += int(predicted_evidence == expected_evidence)

            predicted_context = {
                entry.node_id
                for entry in result.trace_ledger
                if entry.context_read
            }
            expected_context = _expected_context_nodes(batch)
            context_tp += len(predicted_context & expected_context)
            context_fp += len(predicted_context - expected_context)
            context_fn += len(expected_context - predicted_context)

            selected_by_round: dict[int, set[int]] = {}
            trace_valid = True
            for entry in result.trace_ledger:
                selected_count += 1
                local_edge = entry.edge_id
                if not 0 <= local_edge < len(case.edges):
                    trace_valid = False
                    continue
                selected_by_round.setdefault(entry.round_index, set()).add(
                    local_edge
                )
                semantic_invalid += int(not case.edges[local_edge].valid)
            trace_valid_count += int(trace_valid)
            if case.answerable:
                answerable_count += 1
                valid_paths += int(_valid_wavefront(batch, selected_by_round))

            arcs_scored += result.arcs_scored
            contexts_read += result.contexts_read
            rounds += result.rounds
            nodes_expanded += len(result.trace_ledger)

    invariance = _invariance_metrics(
        model,
        batches,
        permuted_batches,
        controller,
        sample_limit=invariance_sample_limit,
    )
    if batches[0].device.type == "cuda":
        torch.cuda.synchronize(batches[0].device)
        peak_memory = torch.cuda.max_memory_allocated(batches[0].device)
    else:
        peak_memory = 0
    runtime = time.perf_counter() - started
    model.train(was_training)

    evidence_precision = _safe_div(evidence_tp, evidence_tp + evidence_fp)
    evidence_recall = _safe_div(evidence_tp, evidence_tp + evidence_fn)
    evidence_f1 = (
        2 * evidence_precision * evidence_recall
        / max(1e-12, evidence_precision + evidence_recall)
    )
    coverage = _safe_div(answered, len(batches))
    rollout = {
        "termination_accuracy": _safe_div(
            termination_correct,
            len(batches),
        ),
        "unknown_reason_accuracy": _safe_div(
            unknown_reason_correct,
            unknown_count,
        ),
        "answered_coverage": coverage,
        "risk_among_answered": _safe_div(answered_errors, answered),
        "false_answer_rate": _safe_div(false_answers, len(batches)),
        "false_unknown_rate": _safe_div(false_unknowns, len(batches)),
        "risk_coverage_operating_point": {
            "coverage": coverage,
            "risk": _safe_div(answered_errors, answered),
        },
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "evidence_f1": evidence_f1,
        "exact_evidence_set_accuracy": _safe_div(
            exact_evidence,
            len(batches),
        ),
        "context_read_precision": _safe_div(
            context_tp,
            context_tp + context_fp,
        ),
        "context_read_recall": _safe_div(
            context_tp,
            context_tp + context_fn,
        ),
        "useful_read_recall": _safe_div(
            context_tp,
            context_tp + context_fn,
        ),
        "unnecessary_reads_per_query": _safe_div(
            context_fp,
            len(batches),
        ),
        "exact_valid_path_rate": _safe_div(valid_paths, answerable_count),
        "trace_validity": _safe_div(trace_valid_count, len(batches)),
        "semantic_invalid_expansion_rate": _safe_div(
            semantic_invalid,
            selected_count,
        ),
    }
    efficiency = {
        "mean_nodes_expanded": _safe_div(nodes_expanded, len(batches)),
        "mean_arcs_scored": _safe_div(arcs_scored, len(batches)),
        "mean_contexts_read": _safe_div(contexts_read, len(batches)),
        "mean_recurrent_rounds": _safe_div(rounds, len(batches)),
        "mean_wall_clock_latency_seconds": _safe_div(
            sum(per_case_latency),
            len(per_case_latency),
        ),
        "peak_cuda_memory_bytes": peak_memory,
    }
    return EvaluationReport(
        split=split,
        case_count=len(batches),
        oracle_loss=oracle_loss,
        teacher_forced=teacher.as_dict(),
        rollout=rollout,
        efficiency=efficiency,
        invariance=invariance,
        runtime_seconds=runtime,
        peak_cuda_memory_bytes=peak_memory,
    )
