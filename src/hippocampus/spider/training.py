from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from ..programs import (
    GeneratorConfig,
    GraphProgramCase,
    GraphProgramGenerator,
    PackedProgramBatch,
    ProgramFamily,
    SyntheticManifoldRenderer,
    pack_rendered_cases,
)
from ..programs.schema import TerminationDecision
from .config import SparseControllerConfig
from .controller import (
    ActionSchedule,
    ActionSource,
    ControllerState,
    SparseWavefrontController,
)
from .execution import (
    ControllerExecutionPolicy,
    HorizonMode,
    apply_path_state_intervention,
)
from .losses import (
    CandidateSupervision,
    LossTerm,
    SpiderLossConfig,
    SpiderLossReport,
    candidate_loss_report,
    termination_loss_report,
)
from .model import CandidateScorerBase
from .state_oracle import StateOracle
from .types import CandidateOutputs


TERMINATION_TO_INDEX = {
    TerminationDecision.CONTINUE: 0,
    TerminationDecision.ANSWER: 1,
    TerminationDecision.UNKNOWN_ABSENT: 2,
    TerminationDecision.UNKNOWN_CONFLICT: 3,
    TerminationDecision.UNKNOWN_INCOMPLETE: 4,
    TerminationDecision.UNKNOWN_UNSUPPORTED: 5,
}


@dataclass(frozen=True, slots=True)
class TrainingLoopConfig:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    seed: int = 0
    log_every: int = 25
    max_grad_norm: float = 5.0
    oracle_fraction_schedule: tuple[float, ...] = (1.0,)
    action_schedule: tuple[ActionSchedule, ...] = ()

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not self.oracle_fraction_schedule:
            raise ValueError("oracle_fraction_schedule may not be empty")
        if any(
            fraction < 0.0 or fraction > 1.0
            for fraction in self.oracle_fraction_schedule
        ):
            raise ValueError("oracle fractions must be in [0, 1]")

    @property
    def resolved_action_schedule(self) -> tuple[ActionSchedule, ...]:
        if self.action_schedule:
            return self.action_schedule
        return tuple(
            ActionSchedule(fraction, fraction, fraction, fraction)
            for fraction in self.oracle_fraction_schedule
        )


@dataclass(frozen=True, slots=True)
class OracleMetrics:
    candidate_count: int
    expand_correct: int
    context_correct: int
    evidence_correct: int
    support_correct: int
    conflict_correct: int
    priority_opportunities: int
    priority_top1_correct: int
    priority_reciprocal_rank_sum: float
    frontier_recall_at_4_sum: float
    predicted_expansions: int
    invalid_expansions: int
    context_true_positive: int
    context_false_positive: int
    context_false_negative: int
    evidence_true_positive: int
    evidence_false_positive: int
    evidence_false_negative: int
    evidence_positive_labels: int
    evidence_negative_labels: int
    termination_count: int
    termination_correct: int

    @property
    def expand_accuracy(self) -> float:
        return self.expand_correct / max(1, self.candidate_count)

    @property
    def context_accuracy(self) -> float:
        return self.context_correct / max(1, self.candidate_count)

    @property
    def evidence_accuracy(self) -> float:
        return self.evidence_correct / max(1, self.candidate_count)

    @property
    def support_accuracy(self) -> float:
        return self.support_correct / max(1, self.candidate_count)

    @property
    def conflict_accuracy(self) -> float:
        return self.conflict_correct / max(1, self.candidate_count)

    @property
    def priority_top1(self) -> float:
        return self.priority_top1_correct / max(1, self.priority_opportunities)

    @property
    def candidate_mrr(self) -> float:
        return self.priority_reciprocal_rank_sum / max(
            1,
            self.priority_opportunities,
        )

    @property
    def valid_frontier_recall_at_4(self) -> float:
        return self.frontier_recall_at_4_sum / max(
            1,
            self.priority_opportunities,
        )

    @property
    def invalid_expansion_rate(self) -> float:
        return self.invalid_expansions / max(1, self.predicted_expansions)

    @property
    def context_precision(self) -> float:
        return self.context_true_positive / max(
            1,
            self.context_true_positive + self.context_false_positive,
        )

    @property
    def context_recall(self) -> float:
        return self.context_true_positive / max(
            1,
            self.context_true_positive + self.context_false_negative,
        )

    @property
    def evidence_precision(self) -> float:
        return self.evidence_true_positive / max(
            1,
            self.evidence_true_positive + self.evidence_false_positive,
        )

    @property
    def evidence_recall(self) -> float:
        return self.evidence_true_positive / max(
            1,
            self.evidence_true_positive + self.evidence_false_negative,
        )

    @property
    def evidence_f1(self) -> float:
        precision = self.evidence_precision
        recall = self.evidence_recall
        return 2 * precision * recall / max(1e-12, precision + recall)

    @property
    def termination_accuracy(self) -> float:
        return self.termination_correct / max(1, self.termination_count)

    @property
    def joint_action_accuracy(self) -> float:
        return min(
            self.expand_accuracy,
            self.context_accuracy,
            self.evidence_accuracy,
            self.termination_accuracy,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "candidate_count": self.candidate_count,
            "candidate_expand_accuracy": self.expand_accuracy,
            "context_read_accuracy": self.context_accuracy,
            "evidence_inclusion_accuracy": self.evidence_accuracy,
            "support_accuracy": self.support_accuracy,
            "conflict_accuracy": self.conflict_accuracy,
            "candidate_top1": self.priority_top1,
            "candidate_mrr": self.candidate_mrr,
            "valid_frontier_recall_at_4": self.valid_frontier_recall_at_4,
            "invalid_expansion_rate": self.invalid_expansion_rate,
            "context_read_precision": self.context_precision,
            "context_read_recall": self.context_recall,
            "evidence_precision": self.evidence_precision,
            "evidence_recall": self.evidence_recall,
            "evidence_f1": self.evidence_f1,
            "evidence_positive_labels": self.evidence_positive_labels,
            "evidence_negative_labels": self.evidence_negative_labels,
            "termination_accuracy": self.termination_accuracy,
            "joint_action_accuracy": self.joint_action_accuracy,
        }

    def __add__(self, other: "OracleMetrics") -> "OracleMetrics":
        return OracleMetrics(
            *(
                getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            )
        )

    @classmethod
    def empty(cls) -> "OracleMetrics":
        return cls(*(0 for _ in cls.__dataclass_fields__))


@dataclass(frozen=True, slots=True)
class OracleRollout:
    loss: torch.Tensor
    report: SpiderLossReport
    metrics: OracleMetrics
    rounds: int
    diagnostics: tuple["RolloutRoundDiagnostic", ...] = ()
    evidence_scores: torch.Tensor | None = None
    evidence_labels: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class RolloutRoundDiagnostic:
    round_index: int
    frontier_source: ActionSource
    context_source: ActionSource
    evidence_source: ActionSource
    termination_source: ActionSource
    frontier_count: int
    context_count: int
    evidence_count: int
    termination_hypothesis_count: int
    termination_target: TerminationDecision
    executed_termination: TerminationDecision


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    step: int
    loss: float
    gradient_norm: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    records: tuple[TrainingRecord, ...]
    initial_metrics: OracleMetrics
    final_metrics: OracleMetrics
    runtime_seconds: float
    action_source_counts: dict[str, int]
    unique_cases_seen: int
    training_examples: int


def _round_metrics(
    outputs: CandidateOutputs,
    context_logits: torch.Tensor,
    supervision: CandidateSupervision,
    frontier_positions: torch.Tensor,
    termination_logits: torch.Tensor,
    termination_target: int,
    *,
    frontier_count: int,
) -> OracleMetrics:
    count = supervision.count
    expand_correct = int(
        (
            (outputs.expand_logits >= 0) == supervision.acceptable
        ).sum().item()
    )
    context_correct = int(
        (
            (context_logits >= 0) == supervision.context_has_value
        ).sum().item()
    )
    evidence_correct = int(
        (
            (outputs.evidence_logits >= 0)
            == supervision.include_as_evidence
        ).sum().item()
    )
    support_correct = int(
        (
            (outputs.support_logits >= 0)
            == supervision.support.bool()
        ).sum().item()
    )
    conflict_correct = int(
        (
            (outputs.conflict_logits >= 0)
            == supervision.conflict.bool()
        ).sum().item()
    )
    priority_count = 0
    priority_correct = 0
    reciprocal_rank_sum = 0.0
    recall_at_4_sum = 0.0
    for frontier in range(frontier_count):
        positions = torch.nonzero(
            frontier_positions == frontier,
            as_tuple=False,
        ).flatten()
        if positions.numel() == 0:
            continue
        positives = positions[supervision.acceptable[positions]]
        if positives.numel() == 0:
            continue
        priority_count += 1
        order = positions[
            torch.argsort(
                outputs.priority_logits[positions],
                descending=True,
                stable=True,
            )
        ]
        winner = order[0]
        priority_correct += int(bool((positives == winner).any().item()))
        positive_ranks = torch.nonzero(
            supervision.acceptable[order],
            as_tuple=False,
        ).flatten()
        reciprocal_rank_sum += 1.0 / (int(positive_ranks[0].item()) + 1)
        recall_at_4_sum += float(
            supervision.acceptable[order[:4]].sum().item()
        ) / positives.numel()
    expand_predictions = outputs.expand_logits >= 0
    context_predictions = context_logits >= 0
    evidence_predictions = outputs.evidence_logits >= 0
    termination_prediction = int(termination_logits.argmax(dim=-1)[0].item())
    return OracleMetrics(
        candidate_count=count,
        expand_correct=expand_correct,
        context_correct=context_correct,
        evidence_correct=evidence_correct,
        support_correct=support_correct,
        conflict_correct=conflict_correct,
        priority_opportunities=priority_count,
        priority_top1_correct=priority_correct,
        priority_reciprocal_rank_sum=reciprocal_rank_sum,
        frontier_recall_at_4_sum=recall_at_4_sum,
        predicted_expansions=int(expand_predictions.sum().item()),
        invalid_expansions=int(
            (expand_predictions & ~supervision.acceptable).sum().item()
        ),
        context_true_positive=int(
            (
                context_predictions & supervision.context_has_value
            ).sum().item()
        ),
        context_false_positive=int(
            (
                context_predictions & ~supervision.context_has_value
            ).sum().item()
        ),
        context_false_negative=int(
            (
                ~context_predictions & supervision.context_has_value
            ).sum().item()
        ),
        evidence_true_positive=int(
            (
                evidence_predictions & supervision.include_as_evidence
            ).sum().item()
        ),
        evidence_false_positive=int(
            (
                evidence_predictions & ~supervision.include_as_evidence
            ).sum().item()
        ),
        evidence_false_negative=int(
            (
                ~evidence_predictions & supervision.include_as_evidence
            ).sum().item()
        ),
        evidence_positive_labels=int(
            supervision.include_as_evidence.sum().item()
        ),
        evidence_negative_labels=int(
            (~supervision.include_as_evidence).sum().item()
        ),
        termination_count=1,
        termination_correct=int(termination_prediction == termination_target),
    )


def _aggregate_reports(
    reports: Sequence[SpiderLossReport],
    *,
    reference: torch.Tensor,
) -> SpiderLossReport:
    names = tuple(reports[0].terms) if reports else ()
    aggregated: dict[str, LossTerm] = {}
    for name in names:
        terms = [report.terms[name] for report in reports]
        count = sum(term.target_count for term in terms)
        if count:
            raw = sum(
                term.raw * term.target_count for term in terms
            ) / count
        else:
            raw = reference.sum() * 0.0
        first = terms[0]
        if bool((first.raw.detach().abs() > 0).item()):
            weight = first.weighted.detach() / first.raw.detach()
            weighted = raw * weight
        else:
            weighted = sum(term.weighted for term in terms) / max(1, len(terms))
        aggregated[name] = LossTerm(raw, weighted, count)
    return SpiderLossReport(aggregated)


def controller_rollout(
    model: CandidateScorerBase,
    batch: PackedProgramBatch,
    *,
    controller_config: SparseControllerConfig,
    action_schedule: ActionSchedule,
    randomizer: random.Random,
    loss_config: SpiderLossConfig | None = None,
    max_rounds: int | None = None,
    execution_policy: ControllerExecutionPolicy | None = None,
) -> OracleRollout:
    """Train on the exact state machine used by autonomous execution."""

    if batch.graph_count != 1:
        raise ValueError("closed-loop rollout currently requires singleton batches")
    config = loss_config or SpiderLossConfig()
    controller = SparseWavefrontController(controller_config)
    policy = execution_policy or ControllerExecutionPolicy.learned()
    oracle = StateOracle(batch.cases[0], batch, controller_config)
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    state = ControllerState.initial()
    reports: list[SpiderLossReport] = []
    metrics = OracleMetrics.empty()
    diagnostics: list[RolloutRoundDiagnostic] = []
    evidence_scores: list[torch.Tensor] = []
    evidence_labels: list[torch.Tensor] = []
    configured_round_limit = max_rounds or controller_config.max_rounds
    round_limit = policy.resolve_round_limit(
        batch,
        configured_max_rounds=configured_round_limit,
    )

    for round_offset in range(round_limit):
        hypotheses = apply_path_state_intervention(
            model,
            batch,
            hypotheses,
            intervention=policy.path_state_intervention,
            round_index=state.round_index,
            seed=policy.intervention_seed,
        )
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
            schedule=action_schedule,
            randomizer=randomizer,
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
        termination_target = oracle.termination_target(transition)
        target_index = TERMINATION_TO_INDEX[termination_target.decision]
        termination_output = model.termination_output(
            batch,
            transition.next_hypotheses,
            transition.next_evidence,
            transition.termination_control,
        )
        termination_logits = termination_output.logits
        candidate_report = candidate_loss_report(
            transition.refined_outputs,
            supervision.candidates,
            proposal.expansion.frontier_positions,
            frontier_count=hypotheses.count,
            config=config,
            context_logits=proposal.candidate_outputs.context_logits,
        )
        termination_report = termination_loss_report(
            termination_output,
            torch.tensor(
                [target_index],
                dtype=torch.int64,
                device=batch.device,
            ),
            config=config,
        )
        reports.append(
            SpiderLossReport(
                terms={
                    **candidate_report.terms,
                    **termination_report.terms,
                }
            )
        )
        metrics = metrics + _round_metrics(
            transition.refined_outputs,
            proposal.candidate_outputs.context_logits,
            supervision.candidates,
            proposal.expansion.frontier_positions,
            termination_logits,
            target_index,
            frontier_count=hypotheses.count,
        )
        evidence_scores.append(
            transition.refined_outputs.evidence_logits.detach()
        )
        evidence_labels.append(
            supervision.candidates.include_as_evidence.detach()
        )
        model_decision = controller.execute_termination(
            termination_logits,
            transition,
        )[0]
        if (
            policy.horizon_mode is not HorizonMode.LEARNED
            and round_offset < round_limit - 1
        ):
            executed = TerminationDecision.CONTINUE
        else:
            executed = (
                termination_target.decision
                if actions.termination_source is ActionSource.ORACLE
                else model_decision
            )
        diagnostics.append(
            RolloutRoundDiagnostic(
                round_index=state.round_index,
                frontier_source=actions.frontier_source,
                context_source=actions.context_source,
                evidence_source=actions.evidence_source,
                termination_source=actions.termination_source,
                frontier_count=int(
                    actions.frontier_candidate_indices.numel()
                ),
                context_count=int(
                    actions.context_candidate_indices.numel()
                ),
                evidence_count=int(
                    actions.evidence_candidate_indices.numel()
                ),
                termination_hypothesis_count=(
                    transition.next_hypotheses.count
                ),
                termination_target=termination_target.decision,
                executed_termination=executed,
            )
        )
        hypotheses = transition.next_hypotheses
        evidence = transition.next_evidence
        state = transition.next_controller_state
        if (
            policy.horizon_mode is HorizonMode.LEARNED
            and executed is not TerminationDecision.CONTINUE
        ):
            break

    report = _aggregate_reports(reports, reference=model.path_seed)
    return OracleRollout(
        loss=report.total,
        report=report,
        metrics=metrics,
        rounds=len(reports),
        diagnostics=tuple(diagnostics),
        evidence_scores=(
            torch.cat(evidence_scores)
            if evidence_scores
            else model.path_seed.new_empty((0,))
        ),
        evidence_labels=(
            torch.cat(evidence_labels)
            if evidence_labels
            else torch.empty(0, dtype=torch.bool, device=batch.device)
        ),
    )


def _default_controller_config(
    batch: PackedProgramBatch,
    *,
    max_rounds: int | None = None,
) -> SparseControllerConfig:
    case = batch.cases[0]
    return SparseControllerConfig(
        max_rounds=max_rounds or max(2, len(case.trace.rounds) + 2),
        frontier_width=32,
        hypotheses_per_node=2,
        context_read_budget=case.context_budget,
        evidence_selection_budget=max(4, len(case.evidence_nodes)),
        search_budget=case.search_budget,
        max_depth=max(12, len(case.trace.rounds) + 2),
    )


def oracle_rollout(
    model: CandidateScorerBase,
    batch: PackedProgramBatch,
    *,
    loss_config: SpiderLossConfig | None = None,
    controller_config: SparseControllerConfig | None = None,
    execution_policy: ControllerExecutionPolicy | None = None,
) -> OracleRollout:
    return controller_rollout(
        model,
        batch,
        controller_config=(
            controller_config or _default_controller_config(batch)
        ),
        action_schedule=ActionSchedule.oracle_only(),
        randomizer=random.Random(0),
        loss_config=loss_config,
        execution_policy=execution_policy,
    )


def mixed_rollout(
    model: CandidateScorerBase,
    batch: PackedProgramBatch,
    *,
    oracle_fraction: float | None = None,
    action_schedule: ActionSchedule | None = None,
    randomizer: random.Random,
    loss_config: SpiderLossConfig | None = None,
    max_rounds: int | None = None,
    controller_config: SparseControllerConfig | None = None,
    execution_policy: ControllerExecutionPolicy | None = None,
) -> OracleRollout:
    """Scheduled execution using the same proposal and transition as runtime."""

    if action_schedule is None:
        if oracle_fraction is None or not 0.0 <= oracle_fraction <= 1.0:
            raise ValueError(
                "provide an action schedule or oracle_fraction in [0, 1]"
            )
        action_schedule = ActionSchedule(
            oracle_fraction,
            oracle_fraction,
            oracle_fraction,
            oracle_fraction,
        )
    resolved_controller = (
        controller_config
        or _default_controller_config(batch, max_rounds=max_rounds)
    )
    return controller_rollout(
        model,
        batch,
        controller_config=resolved_controller,
        action_schedule=action_schedule,
        randomizer=randomizer,
        loss_config=loss_config,
        max_rounds=max_rounds,
        execution_policy=execution_policy,
    )


def make_tiny_cases(
    *,
    case_count: int = 48,
    seed: int = 1101,
    generator_config: GeneratorConfig | None = None,
) -> tuple[GraphProgramCase, ...]:
    if case_count < 8:
        raise ValueError("tiny fixture needs at least eight cases")
    generator = GraphProgramGenerator(
        generator_config
        or GeneratorConfig(
            min_nodes=8,
            max_nodes=14,
            min_path_length=1,
            max_path_length=4,
        )
    )
    families = tuple(ProgramFamily)
    cases: list[GraphProgramCase] = []
    for index in range(case_count):
        family = families[index % len(families)]
        outcome_group = index // len(families)
        answerable = outcome_group % 2 == 0
        unknown_decision = None
        context_budget_exhausted = False
        if not answerable:
            negative_variant = (outcome_group // 2) % 3
            if negative_variant == 1:
                unknown_decision = TerminationDecision.UNKNOWN_INCOMPLETE
                context_budget_exhausted = family is ProgramFamily.LATEST_VALID
            elif negative_variant == 2:
                unknown_decision = TerminationDecision.UNKNOWN_UNSUPPORTED
        cases.append(
            generator.generate(
                family=family,
                seed=seed + index,
                answerable=answerable,
                require_multiple_paths=(
                    family is ProgramFamily.REACHABILITY and index % 3 == 0
                ),
                unknown_decision=unknown_decision,
                context_budget_exhausted=context_budget_exhausted,
            )
        )
    return tuple(cases)


def evaluate_oracle_batches(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    loss_config: SpiderLossConfig | None = None,
    controller_config: SparseControllerConfig | None = None,
    execution_policy: ControllerExecutionPolicy | None = None,
) -> tuple[float, OracleMetrics]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    metrics = OracleMetrics.empty()
    with torch.no_grad():
        for batch in batches:
            result = oracle_rollout(
                model,
                batch,
                loss_config=loss_config,
                controller_config=controller_config,
                execution_policy=execution_policy,
            )
            total_loss += float(result.loss.item())
            metrics = metrics + result.metrics
    model.train(was_training)
    return total_loss / max(1, len(batches)), metrics


def train_oracle_batches(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    loop_config: TrainingLoopConfig,
    loss_config: SpiderLossConfig | None = None,
    controller_config: SparseControllerConfig | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int | None = None,
    execution_policy: ControllerExecutionPolicy | None = None,
) -> TrainingResult:
    if not batches:
        raise ValueError("training requires at least one packed batch")
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    loss_settings = loss_config or SpiderLossConfig()
    randomizer = random.Random(loop_config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=loop_config.learning_rate,
        weight_decay=loop_config.weight_decay,
    )
    initial_loss, initial_metrics = evaluate_oracle_batches(
        model,
        batches,
        loss_config=loss_settings,
        controller_config=controller_config,
        execution_policy=execution_policy,
    )
    records = [
        TrainingRecord(
            step=0,
            loss=initial_loss,
            gradient_norm=0.0,
            elapsed_seconds=0.0,
        )
    ]
    started = time.perf_counter()
    model.train()
    epoch_order = list(range(len(batches)))
    randomizer.shuffle(epoch_order)
    epoch_cursor = 0
    seen_case_ids: set[str] = set()
    training_examples = 0
    action_source_counts = {
        f"{action}/{source.value}": 0
        for action in ("frontier", "context", "evidence", "termination")
        for source in ActionSource
    }

    def checkpoint_payload(
        *,
        step: int,
        final_metrics: OracleMetrics | None,
    ) -> dict[str, object]:
        return {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model.config),
            "loop_config": asdict(loop_config),
            "loss_config": asdict(loss_settings),
            "controller_config": (
                asdict(controller_config)
                if controller_config is not None
                else None
            ),
            "execution_policy": asdict(
                execution_policy
                or ControllerExecutionPolicy.learned()
            ),
            "final_metrics": (
                final_metrics.as_dict()
                if final_metrics is not None
                else None
            ),
            "action_source_counts": action_source_counts,
            "unique_cases_seen": len(seen_case_ids),
            "training_examples": training_examples,
        }

    def draw_batch() -> PackedProgramBatch:
        nonlocal epoch_cursor
        if epoch_cursor == len(epoch_order):
            randomizer.shuffle(epoch_order)
            epoch_cursor = 0
        batch = batches[epoch_order[epoch_cursor]]
        epoch_cursor += 1
        return batch

    for step in range(1, loop_config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        selected = [
            draw_batch()
            for _ in range(loop_config.batch_size)
        ]
        schedules = loop_config.resolved_action_schedule
        schedule_index = min(
            len(schedules) - 1,
            (step - 1)
            * len(schedules)
            // loop_config.steps,
        )
        action_schedule = schedules[schedule_index]
        rollouts = [
            controller_rollout(
                model,
                batch,
                controller_config=(
                    controller_config
                    or _default_controller_config(batch)
                ),
                action_schedule=action_schedule,
                randomizer=randomizer,
                loss_config=loss_settings,
                execution_policy=execution_policy,
            )
            for batch in selected
        ]
        for batch, rollout in zip(selected, rollouts, strict=True):
            seen_case_ids.add(batch.cases[0].case_id)
            training_examples += 1
            for diagnostic in rollout.diagnostics:
                action_source_counts[
                    f"frontier/{diagnostic.frontier_source.value}"
                ] += 1
                action_source_counts[
                    f"context/{diagnostic.context_source.value}"
                ] += 1
                action_source_counts[
                    f"evidence/{diagnostic.evidence_source.value}"
                ] += 1
                action_source_counts[
                    f"termination/{diagnostic.termination_source.value}"
                ] += 1
        loss = torch.stack([rollout.loss for rollout in rollouts]).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            loop_config.max_grad_norm,
        )
        optimizer.step()
        if (
            checkpoint_path is not None
            and checkpoint_every is not None
            and step % checkpoint_every == 0
            and step < loop_config.steps
        ):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            intermediate = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_step_{step:06d}"
                f"{checkpoint_path.suffix}"
            )
            torch.save(
                checkpoint_payload(step=step, final_metrics=None),
                intermediate,
            )
        if step % loop_config.log_every == 0 or step == loop_config.steps:
            records.append(
                TrainingRecord(
                    step=step,
                    loss=float(loss.detach().item()),
                    gradient_norm=float(gradient_norm.detach().item()),
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
    runtime = time.perf_counter() - started
    final_loss, final_metrics = evaluate_oracle_batches(
        model,
        batches,
        loss_config=loss_settings,
        controller_config=controller_config,
        execution_policy=execution_policy,
    )
    records.append(
        TrainingRecord(
            step=loop_config.steps,
            loss=final_loss,
            gradient_norm=records[-1].gradient_norm,
            elapsed_seconds=runtime,
        )
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(
                step=loop_config.steps,
                final_metrics=final_metrics,
            ),
            checkpoint_path,
        )
    return TrainingResult(
        records=tuple(records),
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        runtime_seconds=runtime,
        action_source_counts=action_source_counts,
        unique_cases_seen=len(seen_case_ids),
        training_examples=training_examples,
    )
