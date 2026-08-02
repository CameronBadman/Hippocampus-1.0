from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence

from .leakage import MetadataLeakageReport, metadata_leakage_report
from .oracle import verify_case
from .schema import GraphProgramCase, ProgramFamily, TerminationDecision


@dataclass(frozen=True, slots=True)
class ProgramLabelAudit:
    case_count: int
    invalid_case_count: int
    invalid_case_ids: tuple[str, ...]
    evidence_label_mismatch_count: int
    evidence_label_mismatch_case_ids: tuple[str, ...]
    positive_summary_identity_rate: float
    lookup_observable_rule_accuracy: float
    lookup_answerable_case_count: int
    unsupported_case_count: int
    unsupported_unique_symbol_count: int
    unsupported_symbol_reuse_count: int
    query_cardinality_answerability_accuracy: float
    query_cardinality_majority_accuracy: float
    metadata_leakage: MetadataLeakageReport
    termination_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["metadata_leakage"] = asdict(self.metadata_leakage)
        return result


def _symbols(atoms) -> set[str]:
    return {
        symbol
        for atom in atoms
        for symbol in atom.symbols
    }


def _evidence_labels_match(case: GraphProgramCase) -> bool:
    labelled = [
        candidate
        for round_ in case.trace.rounds
        for candidate in round_.candidates
        if candidate.include_as_evidence
    ]
    if case.evidence_edge_ids:
        return {candidate.edge_id for candidate in labelled} == set(
            case.evidence_edge_ids
        )
    return {candidate.destination_node for candidate in labelled} == set(
        case.evidence_nodes
    )


def _lookup_observable_rule(case: GraphProgramCase) -> bool:
    query_symbols = _symbols(case.query_atoms)
    predicted: set[int] = set()
    for round_ in case.trace.rounds:
        for candidate in round_.candidates:
            edge = case.edges[candidate.edge_id]
            destination = case.nodes[candidate.destination_node]
            relation_match = bool(query_symbols & _symbols(edge.atoms))
            value_match = bool(
                query_symbols & _symbols(destination.summary_atoms)
            )
            invalid_gate = any(
                atom.scalar is not None and atom.scalar < 0
                for atom in edge.atoms
            )
            if relation_match and value_match and not invalid_gate:
                predicted.add(candidate.destination_node)
    return predicted == set(case.evidence_nodes)


def _query_cardinality_accuracy(
    cases: Sequence[GraphProgramCase],
) -> tuple[float, float]:
    labels = [case.answerable for case in cases]
    majority = max(sum(labels), len(labels) - sum(labels)) / len(labels)
    groups: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for case in cases:
        groups[(case.family.value, len(case.query_atoms))].append(
            case.answerable
        )
    predictions = {
        key: sum(values) * 2 >= len(values)
        for key, values in groups.items()
    }
    correct = sum(
        predictions[(case.family.value, len(case.query_atoms))]
        == case.answerable
        for case in cases
    )
    return correct / len(cases), majority


def audit_aligned_program_labels(
    cases: Sequence[GraphProgramCase],
) -> ProgramLabelAudit:
    """Audit supervisor consistency and observable identity contracts."""

    if not cases:
        raise ValueError("program label audit requires cases")
    invalid_ids = tuple(
        case.case_id for case in cases if not verify_case(case).valid
    )
    mismatch_ids = tuple(
        case.case_id for case in cases if not _evidence_labels_match(case)
    )
    positive_identity_total = 0
    positive_identity_matches = 0
    for case in cases:
        query_symbols = _symbols(case.query_atoms)
        for node_id in case.evidence_nodes:
            positive_identity_total += 1
            positive_identity_matches += int(
                bool(query_symbols & _symbols(case.nodes[node_id].summary_atoms))
            )
    lookup_cases = [
        case
        for case in cases
        if case.family is ProgramFamily.LOOKUP and case.answerable
    ]
    unsupported_cases = [
        case
        for case in cases
        if case.termination.decision
        is TerminationDecision.UNKNOWN_UNSUPPORTED
    ]
    unsupported_symbols = [
        symbol
        for case in unsupported_cases
        for atom in case.query_atoms
        for symbol in atom.symbols
        if symbol.startswith("unsupported_")
    ]
    cardinality_accuracy, cardinality_majority = (
        _query_cardinality_accuracy(cases)
    )
    return ProgramLabelAudit(
        case_count=len(cases),
        invalid_case_count=len(invalid_ids),
        invalid_case_ids=invalid_ids,
        evidence_label_mismatch_count=len(mismatch_ids),
        evidence_label_mismatch_case_ids=mismatch_ids,
        positive_summary_identity_rate=(
            positive_identity_matches / max(1, positive_identity_total)
        ),
        lookup_observable_rule_accuracy=(
            sum(_lookup_observable_rule(case) for case in lookup_cases)
            / max(1, len(lookup_cases))
        ),
        lookup_answerable_case_count=len(lookup_cases),
        unsupported_case_count=len(unsupported_cases),
        unsupported_unique_symbol_count=len(set(unsupported_symbols)),
        unsupported_symbol_reuse_count=(
            len(unsupported_symbols) - len(set(unsupported_symbols))
        ),
        query_cardinality_answerability_accuracy=cardinality_accuracy,
        query_cardinality_majority_accuracy=cardinality_majority,
        metadata_leakage=metadata_leakage_report(cases),
        termination_counts=dict(
            sorted(
                Counter(
                    case.termination.decision.value for case in cases
                ).items()
            )
        ),
    )
