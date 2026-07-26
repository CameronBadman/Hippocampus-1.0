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
from ..programs.schema import CandidateTarget, TerminationDecision
from .hypothesis import HypothesisBatch
from .losses import (
    CandidateSupervision,
    LossTerm,
    SpiderLossConfig,
    SpiderLossReport,
    candidate_loss_report,
    termination_loss_term,
)
from .model import CandidateScorerBase
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

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")


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


def _target_map(
    candidates: tuple[CandidateTarget, ...],
) -> dict[tuple[int, int, int], CandidateTarget]:
    return {
        (
            candidate.edge_id,
            candidate.source_node,
            candidate.destination_node,
        ): candidate
        for candidate in candidates
    }


def candidate_supervision(
    batch: PackedProgramBatch,
    round_index: int,
    expansion,
) -> CandidateSupervision:
    if batch.graph_count != 1:
        raise ValueError("oracle rollout currently consumes singleton packed cases")
    case = batch.cases[0]
    node_offset = int(batch.graph.topology.graph_node_ptr[0].item())
    edge_offset = int(batch.graph.topology.graph_edge_ptr[0].item())
    targets = _target_map(case.trace.rounds[round_index].candidates)
    resolved: list[CandidateTarget] = []
    for edge_id, source, destination in zip(
        expansion.edge_ids.tolist(),
        expansion.source_node_ids.tolist(),
        expansion.destination_node_ids.tolist(),
        strict=True,
    ):
        key = (
            edge_id - edge_offset,
            source - node_offset,
            destination - node_offset,
        )
        try:
            resolved.append(targets[key])
        except KeyError as exc:
            raise ValueError(
                f"expanded candidate {key} is absent from oracle round {round_index}"
            ) from exc
    device = batch.device
    return CandidateSupervision(
        acceptable=torch.tensor(
            [target.acceptable for target in resolved],
            dtype=torch.bool,
            device=device,
        ),
        context_has_value=torch.tensor(
            [target.context_has_value for target in resolved],
            dtype=torch.bool,
            device=device,
        ),
        include_as_evidence=torch.tensor(
            [target.include_as_evidence for target in resolved],
            dtype=torch.bool,
            device=device,
        ),
        remaining_cost=torch.tensor(
            [target.remaining_cost for target in resolved],
            dtype=torch.float32,
            device=device,
        ),
        support=torch.tensor(
            [target.support for target in resolved],
            dtype=torch.float32,
            device=device,
        ),
        conflict=torch.tensor(
            [target.conflict for target in resolved],
            dtype=torch.float32,
            device=device,
        ),
    )


def _next_oracle_hypotheses(
    model: CandidateScorerBase,
    hypotheses: HypothesisBatch,
    expansion,
    outputs: CandidateOutputs,
    acceptable: torch.Tensor,
) -> HypothesisBatch:
    selected = torch.nonzero(acceptable, as_tuple=False).flatten()
    if selected.numel() == 0:
        return model.empty_hypotheses(hypotheses.device)
    parents = expansion.frontier_positions[selected].to(torch.int64)
    return HypothesisBatch(
        node_ids=expansion.destination_node_ids[selected],
        graph_ids=hypotheses.graph_ids[parents],
        path_state=outputs.next_path_state[selected],
        scores=hypotheses.scores[parents] + outputs.priority_logits[selected],
        depths=hypotheses.depths[parents] + 1,
        parent_trace_ids=torch.full(
            (selected.numel(),),
            -1,
            dtype=torch.int64,
            device=hypotheses.device,
        ),
        incoming_arc_ids=expansion.arc_ids[selected],
        incoming_edge_ids=expansion.edge_ids[selected],
        context_read=torch.zeros(
            selected.numel(),
            dtype=torch.bool,
            device=hypotheses.device,
        ),
    ).validate()


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
        winner = positions[
            outputs.priority_logits[positions].argmax()
        ]
        priority_correct += int(bool((positives == winner).any().item()))
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


def oracle_rollout(
    model: CandidateScorerBase,
    batch: PackedProgramBatch,
    *,
    loss_config: SpiderLossConfig | None = None,
) -> OracleRollout:
    if batch.graph_count != 1:
        raise ValueError("oracle rollout currently requires one graph per batch")
    config = loss_config or SpiderLossConfig()
    hypotheses = model.initial_hypotheses(batch)
    evidence = model.initial_evidence(batch)
    reports: list[SpiderLossReport] = []
    metrics = OracleMetrics.empty()
    case = batch.cases[0]
    for round_index, oracle_round in enumerate(case.trace.rounds):
        expansion = batch.graph.expand_frontier(
            hypotheses.node_ids,
            validate_ids=False,
        )
        supervision = candidate_supervision(batch, round_index, expansion)
        outputs = model.score_candidates(
            batch,
            hypotheses,
            expansion,
            evidence,
            round_index=round_index,
        )
        context_logits = outputs.context_logits
        context_candidates = torch.nonzero(
            supervision.context_has_value,
            as_tuple=False,
        ).flatten()
        if context_candidates.numel():
            outputs = model.refine_with_context(
                batch,
                expansion,
                outputs,
                context_candidates,
            )
        candidate_report = candidate_loss_report(
            outputs,
            supervision,
            expansion.frontier_positions,
            frontier_count=hypotheses.count,
            config=config,
            context_logits=context_logits,
        )
        next_hypotheses = _next_oracle_hypotheses(
            model,
            hypotheses,
            expansion,
            outputs,
            supervision.acceptable,
        )
        evidence_candidates = torch.nonzero(
            supervision.include_as_evidence,
            as_tuple=False,
        ).flatten()
        if evidence_candidates.numel():
            parents = expansion.frontier_positions[
                evidence_candidates
            ].to(torch.int64)
            evidence = model.update_evidence(
                evidence,
                outputs.next_path_state[evidence_candidates].mean(dim=1),
                hypotheses.graph_ids[parents],
            )
        termination_hypotheses = (
            next_hypotheses
            if next_hypotheses.count
            else hypotheses
        )
        termination_logits = model.termination_logits(
            batch,
            termination_hypotheses,
            evidence,
        )
        termination_target = TERMINATION_TO_INDEX[
            oracle_round.termination.decision
        ]
        termination = termination_loss_term(
            termination_logits,
            torch.tensor(
                [termination_target],
                dtype=torch.int64,
                device=batch.device,
            ),
            config=config,
        )
        reports.append(
            SpiderLossReport(
                terms={
                    **candidate_report.terms,
                    "termination": termination,
                }
            )
        )
        metrics = metrics + _round_metrics(
            outputs,
            context_logits,
            supervision,
            expansion.frontier_positions,
            termination_logits,
            termination_target,
            frontier_count=hypotheses.count,
        )
        hypotheses = next_hypotheses
        if round_index + 1 < len(case.trace.rounds) and hypotheses.count == 0:
            raise ValueError("oracle trace continues after an empty acceptable frontier")

    report = _aggregate_reports(
        reports,
        reference=model.path_seed,
    )
    return OracleRollout(
        loss=report.total,
        report=report,
        metrics=metrics,
        rounds=len(case.trace.rounds),
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
    return tuple(
        generator.generate(
            family=families[index % len(families)],
            seed=seed + index,
            answerable=(index // len(families)) % 2 == 0,
            require_multiple_paths=(
                families[index % len(families)]
                is ProgramFamily.REACHABILITY
                and index % 3 == 0
            ),
        )
        for index in range(case_count)
    )


def evaluate_oracle_batches(
    model: CandidateScorerBase,
    batches: Sequence[PackedProgramBatch],
    *,
    loss_config: SpiderLossConfig | None = None,
) -> tuple[float, OracleMetrics]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    metrics = OracleMetrics.empty()
    with torch.no_grad():
        for batch in batches:
            result = oracle_rollout(model, batch, loss_config=loss_config)
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
    checkpoint_path: Path | None = None,
) -> TrainingResult:
    if not batches:
        raise ValueError("training requires at least one packed batch")
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
    for step in range(1, loop_config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        selected = [
            batches[randomizer.randrange(len(batches))]
            for _ in range(loop_config.batch_size)
        ]
        loss = torch.stack(
            [
                oracle_rollout(
                    model,
                    batch,
                    loss_config=loss_settings,
                ).loss
                for batch in selected
            ]
        ).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            loop_config.max_grad_norm,
        )
        optimizer.step()
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
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model.config),
                "loop_config": asdict(loop_config),
                "loss_config": asdict(loss_settings),
                "final_metrics": final_metrics.as_dict(),
            },
            checkpoint_path,
        )
    return TrainingResult(
        records=tuple(records),
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        runtime_seconds=runtime,
    )
