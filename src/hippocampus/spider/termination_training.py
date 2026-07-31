from __future__ import annotations

from dataclasses import dataclass, replace
import random
import time
from typing import Sequence

import torch
from torch.nn import functional as F

from ..programs.batching import PackedProgramBatch
from ..programs.schema import TerminationDecision
from .config import SparseControllerConfig
from .controller import (
    ActionSchedule,
    ControllerState,
    SparseWavefrontController,
    termination_control_features,
)
from .losses import SpiderLossConfig, termination_loss_report
from .model import CandidateScorerBase
from .state_oracle import StateOracle
from .terminator import (
    TerminationFactorTargets,
    TerminationOutput,
)


_DECISION_TO_INDEX = {
    TerminationDecision.CONTINUE: 0,
    TerminationDecision.ANSWER: 1,
    TerminationDecision.UNKNOWN_ABSENT: 2,
    TerminationDecision.UNKNOWN_CONFLICT: 3,
    TerminationDecision.UNKNOWN_INCOMPLETE: 4,
    TerminationDecision.UNKNOWN_UNSUPPORTED: 5,
}


@dataclass(frozen=True, slots=True)
class NullExpansionStateDataset:
    query: torch.Tensor
    evidence: torch.Tensor
    path: torch.Tensor
    control: torch.Tensor
    targets: torch.Tensor
    case_ids: tuple[str, ...]
    round_indices: tuple[int, ...]

    @property
    def count(self) -> int:
        return int(self.targets.numel())

    def validate(self) -> "NullExpansionStateDataset":
        count = self.count
        if self.targets.shape != (count,) or self.targets.dtype != torch.bool:
            raise ValueError("NULL targets must be one-dimensional booleans")
        for name in ("query", "evidence", "path", "control"):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != count:
                raise ValueError(f"{name} must align with NULL targets")
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError(f"{name} must be detached on CPU")
        if len(self.case_ids) != count or len(self.round_indices) != count:
            raise ValueError("NULL metadata must align with targets")
        return self


@dataclass(frozen=True, slots=True)
class TerminationStateDataset:
    query: torch.Tensor
    evidence: torch.Tensor
    frontier: torch.Tensor
    control: torch.Tensor
    decisions: torch.Tensor
    factor_targets: TerminationFactorTargets
    exact_stop: torch.Tensor
    case_ids: tuple[str, ...]
    families: tuple[str, ...]
    round_indices: tuple[int, ...]
    action_schedules: tuple[str, ...]
    null_states: NullExpansionStateDataset | None = None

    @property
    def count(self) -> int:
        return int(self.decisions.numel())

    def validate(self) -> "TerminationStateDataset":
        count = self.count
        if self.decisions.shape != (count,):
            raise ValueError("termination decisions must be one-dimensional")
        if self.decisions.dtype != torch.int64:
            raise TypeError("termination decisions must use int64")
        if self.exact_stop.shape != (count,) or self.exact_stop.dtype != torch.bool:
            raise ValueError("exact_stop must be bool[state_count]")
        for name in ("query", "evidence", "frontier", "control"):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != count:
                raise ValueError(f"{name} must align with termination states")
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError(f"{name} must be detached on CPU")
        self.factor_targets.validate(
            batch_size=count,
            device=torch.device("cpu"),
        )
        metadata = (
            self.case_ids,
            self.families,
            self.round_indices,
            self.action_schedules,
        )
        if any(len(values) != count for values in metadata):
            raise ValueError("termination metadata must align with states")
        if self.null_states is not None:
            self.null_states.validate()
        return self


@dataclass(frozen=True, slots=True)
class HeadTrainingRecord:
    step: int
    loss: float
    gradient_norm: float


@dataclass(frozen=True, slots=True)
class HeadTrainingResult:
    completed_steps: int
    runtime_seconds: float
    initial_loss: float
    final_loss: float
    records: tuple[HeadTrainingRecord, ...]
    trainable_parameter_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TerminationStateEvaluation:
    loss: float
    count: int
    accuracy: float
    continuation_recall: float
    premature_stop_rate: float
    answer_recall: float
    unknown_macro_recall: float
    factor_accuracy: dict[str, float]
    confusion: tuple[tuple[int, ...], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "count": self.count,
            "accuracy": self.accuracy,
            "continuation_recall": self.continuation_recall,
            "premature_stop_rate": self.premature_stop_rate,
            "answer_recall": self.answer_recall,
            "unknown_macro_recall": self.unknown_macro_recall,
            "factor_accuracy": self.factor_accuracy,
            "confusion": [list(row) for row in self.confusion],
        }


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu")


def _schedule_name(schedule: ActionSchedule) -> str:
    if schedule == ActionSchedule.oracle_only():
        return "oracle"
    if schedule == ActionSchedule.model_only():
        return "model"
    return (
        f"mixed-{schedule.frontier:.2f}-{schedule.context:.2f}-"
        f"{schedule.evidence:.2f}"
    )


def _factor_to_cpu(
    targets: TerminationFactorTargets,
) -> TerminationFactorTargets:
    return TerminationFactorTargets(
        evidence_sufficient=_cpu(targets.evidence_sufficient),
        useful_work_remaining=_cpu(targets.useful_work_remaining),
        answer_supported=_cpu(targets.answer_supported),
        unknown_reason=_cpu(targets.unknown_reason),
        unknown_mask=_cpu(targets.unknown_mask),
    )


def _concat_factors(
    values: Sequence[TerminationFactorTargets],
) -> TerminationFactorTargets:
    return TerminationFactorTargets(
        evidence_sufficient=torch.cat(
            [item.evidence_sufficient for item in values]
        ),
        useful_work_remaining=torch.cat(
            [item.useful_work_remaining for item in values]
        ),
        answer_supported=torch.cat(
            [item.answer_supported for item in values]
        ),
        unknown_reason=torch.cat(
            [item.unknown_reason for item in values]
        ),
        unknown_mask=torch.cat(
            [item.unknown_mask for item in values]
        ),
    )


def _exact_stop(
    state: ControllerState,
    config: SparseControllerConfig,
) -> bool:
    return (
        state.frontier_empty
        or state.search_budget_exhausted
        or state.depth_exhausted
        or state.round_index >= config.max_rounds
    )


def collect_termination_state_dataset(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    controller_config: SparseControllerConfig,
    schedules: Sequence[ActionSchedule],
    seed: int,
    collect_null_states: bool = False,
) -> TerminationStateDataset:
    """Materialize exact, frozen head inputs from closed-loop state rollouts."""

    if not batches or not schedules:
        raise ValueError("state collection requires batches and schedules")
    if any(batch.graph_count != 1 for batch in batches):
        raise ValueError("state collection requires singleton packed batches")
    collection_config = (
        replace(controller_config, expansion_policy="threshold")
        if controller_config.expansion_policy == "learned_null"
        else controller_config
    )
    controller = SparseWavefrontController(collection_config)
    randomizer = random.Random(seed)
    query_rows: list[torch.Tensor] = []
    evidence_rows: list[torch.Tensor] = []
    frontier_rows: list[torch.Tensor] = []
    control_rows: list[torch.Tensor] = []
    decisions: list[torch.Tensor] = []
    factors: list[TerminationFactorTargets] = []
    exact_stops: list[torch.Tensor] = []
    case_ids: list[str] = []
    families: list[str] = []
    round_indices: list[int] = []
    schedule_names: list[str] = []
    null_query: list[torch.Tensor] = []
    null_evidence: list[torch.Tensor] = []
    null_path: list[torch.Tensor] = []
    null_control: list[torch.Tensor] = []
    null_targets: list[torch.Tensor] = []
    null_case_ids: list[str] = []
    null_rounds: list[int] = []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for schedule in schedules:
            schedule_name = _schedule_name(schedule)
            for batch in batches:
                oracle = StateOracle(
                    batch.cases[0],
                    batch,
                    collection_config,
                )
                hypotheses = model.initial_hypotheses(batch)
                evidence = model.initial_evidence(batch)
                state = ControllerState.initial()
                for _ in range(collection_config.max_rounds):
                    proposal = controller.propose(
                        model,
                        batch,
                        hypotheses,
                        evidence,
                        state,
                    )
                    supervision = oracle.label(
                        proposal,
                        hypotheses,
                        state,
                    )
                    if collect_null_states and hypotheses.count:
                        search_limit, context_limit, _ = (
                            controller.resolved_limits(batch)
                        )
                        null_controls = termination_control_features(
                            batch,
                            hypotheses,
                            state,
                            config=collection_config,
                            search_limit=search_limit,
                            context_limit=context_limit,
                        )
                        null_inputs = model.null_expansion_inputs(
                            batch,
                            hypotheses,
                            evidence,
                            null_controls,
                        )
                        branch_targets = torch.ones(
                            hypotheses.count,
                            dtype=torch.bool,
                            device=batch.device,
                        )
                        preserving = (
                            supervision.candidates.acceptable
                            & proposal.depth_eligible
                        )
                        if bool(preserving.any().item()):
                            branch_targets[
                                proposal.expansion.frontier_positions[
                                    preserving
                                ].to(torch.int64)
                            ] = False
                        null_query.append(_cpu(null_inputs[0]))
                        null_evidence.append(_cpu(null_inputs[1]))
                        null_path.append(_cpu(null_inputs[2]))
                        null_control.append(_cpu(null_inputs[3]))
                        null_targets.append(_cpu(branch_targets))
                        null_case_ids.extend(
                            [batch.cases[0].case_id] * hypotheses.count
                        )
                        null_rounds.extend(
                            [state.round_index] * hypotheses.count
                        )
                    selection = controller.select_actions(
                        model,
                        batch,
                        proposal,
                        supervision=supervision,
                        state=state,
                        schedule=schedule,
                        randomizer=randomizer,
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
                    direct = oracle.termination_factor_targets(transition)
                    decision = oracle.termination_target(
                        transition
                    ).decision
                    inputs = model.termination_inputs(
                        batch,
                        transition.next_hypotheses,
                        transition.next_evidence,
                        transition.termination_control,
                    )
                    query_rows.append(_cpu(inputs[0]))
                    evidence_rows.append(_cpu(inputs[1]))
                    frontier_rows.append(_cpu(inputs[2]))
                    control_rows.append(_cpu(inputs[3]))
                    decisions.append(
                        torch.tensor(
                            [_DECISION_TO_INDEX[decision]],
                            dtype=torch.int64,
                        )
                    )
                    factors.append(_factor_to_cpu(direct))
                    exact_stops.append(
                        torch.tensor(
                            [
                                _exact_stop(
                                    transition.next_controller_state,
                                    collection_config,
                                )
                            ],
                            dtype=torch.bool,
                        )
                    )
                    case_ids.append(batch.cases[0].case_id)
                    families.append(batch.cases[0].family.value)
                    round_indices.append(state.round_index)
                    schedule_names.append(schedule_name)
                    hypotheses = transition.next_hypotheses
                    evidence = transition.next_evidence
                    state = transition.next_controller_state
                    if (
                        not direct.useful_work_remaining.item()
                        and (
                            direct.evidence_sufficient.item()
                            or direct.unknown_mask.item()
                        )
                    ):
                        break

    model.train(was_training)
    null_dataset = (
        NullExpansionStateDataset(
            query=torch.cat(null_query),
            evidence=torch.cat(null_evidence),
            path=torch.cat(null_path),
            control=torch.cat(null_control),
            targets=torch.cat(null_targets),
            case_ids=tuple(null_case_ids),
            round_indices=tuple(null_rounds),
        ).validate()
        if collect_null_states and null_targets
        else None
    )
    return TerminationStateDataset(
        query=torch.cat(query_rows),
        evidence=torch.cat(evidence_rows),
        frontier=torch.cat(frontier_rows),
        control=torch.cat(control_rows),
        decisions=torch.cat(decisions),
        factor_targets=_concat_factors(factors),
        exact_stop=torch.cat(exact_stops),
        case_ids=tuple(case_ids),
        families=tuple(families),
        round_indices=tuple(round_indices),
        action_schedules=tuple(schedule_names),
        null_states=null_dataset,
    ).validate()


class _BalancedSampler:
    def __init__(self, keys: Sequence[object], *, seed: int) -> None:
        groups: dict[object, list[int]] = {}
        for index, key in enumerate(keys):
            groups.setdefault(key, []).append(index)
        if not groups:
            raise ValueError("balanced sampling requires examples")
        self.groups = tuple(groups[key] for key in sorted(groups, key=str))
        self.randomizer = random.Random(seed)
        self.cursor = 0

    def draw(self, count: int) -> torch.Tensor:
        indices = []
        for _ in range(count):
            group = self.groups[self.cursor % len(self.groups)]
            indices.append(self.randomizer.choice(group))
            self.cursor += 1
        self.randomizer.shuffle(indices)
        return torch.tensor(indices, dtype=torch.int64)


def _termination_strata(
    dataset: TerminationStateDataset,
) -> tuple[tuple[object, ...], ...]:
    factors = dataset.factor_targets
    return tuple(
        (
            bool(factors.evidence_sufficient[index].item()),
            bool(factors.useful_work_remaining[index].item()),
            bool(factors.answer_supported[index].item()),
            (
                int(factors.unknown_reason[index].item())
                if bool(factors.unknown_mask[index].item())
                else -1
            ),
        )
        for index in range(dataset.count)
    )


def _selected_factors(
    factors: TerminationFactorTargets,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> TerminationFactorTargets:
    return TerminationFactorTargets(
        evidence_sufficient=factors.evidence_sufficient[indices].to(device),
        useful_work_remaining=factors.useful_work_remaining[indices].to(device),
        answer_supported=factors.answer_supported[indices].to(device),
        unknown_reason=factors.unknown_reason[indices].to(device),
        unknown_mask=factors.unknown_mask[indices].to(device),
    )


def _termination_output(
    model: CandidateScorerBase,
    dataset: TerminationStateDataset,
    indices: torch.Tensor,
) -> TerminationOutput:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    raw = model.termination_head(
        dataset.query[indices].to(device=device, dtype=dtype),
        dataset.evidence[indices].to(device=device, dtype=dtype),
        dataset.frontier[indices].to(device=device, dtype=dtype),
        dataset.control[indices].to(device=device, dtype=dtype),
    )
    return (
        TerminationOutput(logits=raw)
        if isinstance(raw, torch.Tensor)
        else raw
    )


def _set_trainable_head(
    model: CandidateScorerBase,
    prefix: str,
) -> tuple[str, ...]:
    names = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefix)
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    if not names:
        raise ValueError(f"model has no parameters under {prefix!r}")
    model.eval()
    module = getattr(model, prefix.removesuffix("."))
    module.train()
    return tuple(names)


def train_frozen_termination_head(
    model: CandidateScorerBase,
    dataset: TerminationStateDataset,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    loss_config: SpiderLossConfig,
    log_every: int = 10,
) -> HeadTrainingResult:
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("termination training settings must be positive")
    dataset.validate()
    trainable = _set_trainable_head(model, "termination_head.")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    sampler = _BalancedSampler(_termination_strata(dataset), seed=seed)
    initial = evaluate_termination_state_dataset(
        model,
        dataset,
        loss_config=loss_config,
    )
    model.termination_head.train()
    records = [HeadTrainingRecord(0, initial.loss, 0.0)]
    started = time.perf_counter()
    for step in range(1, steps + 1):
        indices = sampler.draw(batch_size)
        output = _termination_output(model, dataset, indices)
        report = termination_loss_report(
            output,
            dataset.decisions[indices].to(output.logits.device),
            factor_targets=(
                _selected_factors(
                    dataset.factor_targets,
                    indices,
                    device=output.logits.device,
                )
                if output.evidence_sufficient_logits is not None
                else None
            ),
            config=loss_config,
        )
        optimizer.zero_grad(set_to_none=True)
        report.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if step % log_every == 0 or step == steps:
            records.append(
                HeadTrainingRecord(
                    step=step,
                    loss=float(report.total.detach().item()),
                    gradient_norm=float(gradient_norm.detach().item()),
                )
            )
    final = evaluate_termination_state_dataset(
        model,
        dataset,
        loss_config=loss_config,
    )
    return HeadTrainingResult(
        completed_steps=steps,
        runtime_seconds=time.perf_counter() - started,
        initial_loss=initial.loss,
        final_loss=final.loss,
        records=tuple(records),
        trainable_parameter_names=trainable,
    )


def train_frozen_null_head(
    model: CandidateScorerBase,
    dataset: NullExpansionStateDataset,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    log_every: int = 10,
) -> HeadTrainingResult:
    if model.null_expansion_head is None:
        raise ValueError("model does not have a NULL expansion head")
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("NULL training settings must be positive")
    dataset.validate()
    trainable = _set_trainable_head(model, "null_expansion_head.")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    sampler = _BalancedSampler(
        tuple(bool(value) for value in dataset.targets.tolist()),
        seed=seed,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def loss_for(indices: torch.Tensor) -> torch.Tensor:
        inputs = (
            dataset.query[indices].to(device=device, dtype=dtype),
            dataset.evidence[indices].to(device=device, dtype=dtype),
            dataset.path[indices].to(device=device, dtype=dtype),
            dataset.control[indices].to(device=device, dtype=dtype),
        )
        logits = model.null_expansion_head(
            torch.cat(inputs, dim=-1)
        ).squeeze(-1)
        return F.binary_cross_entropy_with_logits(
            logits.float(),
            dataset.targets[indices].to(device=device).float(),
        )

    with torch.no_grad():
        all_indices = torch.arange(dataset.count, dtype=torch.int64)
        initial_loss = float(loss_for(all_indices).item())
    records = [HeadTrainingRecord(0, initial_loss, 0.0)]
    started = time.perf_counter()
    for step in range(1, steps + 1):
        indices = sampler.draw(batch_size)
        loss = loss_for(indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if step % log_every == 0 or step == steps:
            records.append(
                HeadTrainingRecord(
                    step=step,
                    loss=float(loss.detach().item()),
                    gradient_norm=float(gradient_norm.detach().item()),
                )
            )
    with torch.no_grad():
        final_loss = float(loss_for(all_indices).item())
    return HeadTrainingResult(
        completed_steps=steps,
        runtime_seconds=time.perf_counter() - started,
        initial_loss=initial_loss,
        final_loss=final_loss,
        records=tuple(records),
        trainable_parameter_names=trainable,
    )


def _factorized_predictions(
    output: TerminationOutput,
    exact_stop: torch.Tensor,
) -> torch.Tensor:
    assert output.evidence_sufficient_logits is not None
    assert output.useful_work_remaining_logits is not None
    assert output.answer_supported_logits is not None
    assert output.unknown_logits is not None
    sufficient = output.evidence_sufficient_logits >= 0
    useful = output.useful_work_remaining_logits >= 0
    answer = output.answer_supported_logits >= 0
    stop = sufficient | ~useful | exact_stop.to(output.logits.device)
    predictions = torch.zeros_like(sufficient, dtype=torch.int64)
    answer_mask = stop & sufficient & answer
    predictions[answer_mask] = 1
    unknown_mask = stop & ~answer_mask
    predictions[unknown_mask] = (
        output.unknown_logits[unknown_mask].argmax(dim=-1) + 2
    )
    return predictions


def _recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_index: int,
) -> float:
    mask = targets == class_index
    return (
        float((predictions[mask] == class_index).float().mean().item())
        if bool(mask.any().item())
        else 0.0
    )


def evaluate_termination_state_dataset(
    model: CandidateScorerBase,
    dataset: TerminationStateDataset,
    *,
    loss_config: SpiderLossConfig,
) -> TerminationStateEvaluation:
    dataset.validate()
    indices = torch.arange(dataset.count, dtype=torch.int64)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = _termination_output(model, dataset, indices)
        direct = (
            _selected_factors(
                dataset.factor_targets,
                indices,
                device=output.logits.device,
            )
            if output.evidence_sufficient_logits is not None
            else None
        )
        report = termination_loss_report(
            output,
            dataset.decisions.to(output.logits.device),
            factor_targets=direct,
            config=loss_config,
        )
        predictions = (
            _factorized_predictions(output, dataset.exact_stop)
            if output.evidence_sufficient_logits is not None
            else output.logits.argmax(dim=-1)
        ).to("cpu")
    model.train(was_training)
    targets = dataset.decisions
    confusion = torch.zeros((6, 6), dtype=torch.int64)
    for target, prediction in zip(
        targets.tolist(),
        predictions.tolist(),
        strict=True,
    ):
        confusion[target, prediction] += 1
    continuation_recall = _recall(predictions, targets, 0)
    answer_recall = _recall(predictions, targets, 1)
    unknown_recalls = [
        _recall(predictions, targets, class_index)
        for class_index in range(2, 6)
        if bool((targets == class_index).any().item())
    ]
    factor_accuracy: dict[str, float] = {}
    if output.evidence_sufficient_logits is not None:
        assert direct is not None
        factor_accuracy = {
            "evidence_sufficient": float(
                (
                    (output.evidence_sufficient_logits >= 0)
                    == direct.evidence_sufficient
                ).float().mean().item()
            ),
            "useful_work_remaining": float(
                (
                    (output.useful_work_remaining_logits >= 0)
                    == direct.useful_work_remaining
                ).float().mean().item()
            ),
            "answer_supported": float(
                (
                    (output.answer_supported_logits >= 0)
                    == direct.answer_supported
                ).float().mean().item()
            ),
        }
    return TerminationStateEvaluation(
        loss=float(report.total.item()),
        count=dataset.count,
        accuracy=float((predictions == targets).float().mean().item()),
        continuation_recall=continuation_recall,
        premature_stop_rate=1.0 - continuation_recall,
        answer_recall=answer_recall,
        unknown_macro_recall=(
            sum(unknown_recalls) / len(unknown_recalls)
            if unknown_recalls
            else 0.0
        ),
        factor_accuracy=factor_accuracy,
        confusion=tuple(
            tuple(int(value) for value in row)
            for row in confusion.tolist()
        ),
    )
