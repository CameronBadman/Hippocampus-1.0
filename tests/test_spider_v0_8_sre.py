from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hippocampus.sre import (
    SRECandidate,
    SRECandidateLabel,
    SRERelationship,
    SRERetrievalCase,
    build_candidate_features,
    build_runtime_vocabulary,
    evaluate_retrieval,
    load_sre_cases,
    pack_sre_batch,
    split_sre_development,
)


def _case(index: int, *, pool_size: int = 4, positive_count: int = 1) -> SRERetrievalCase:
    candidates = tuple(
        SRECandidate(
            candidate_id=f"memory-{index}-{candidate}",
            text=f"incident {index} memory {candidate}",
            memory_type="incident" if candidate == 0 else "note",
            occurred_at=f"2026-01-{candidate + 1:02d}T00:00:00Z",
            status="active" if candidate < positive_count else "superseded",
            region="us-east-1",
        )
        for candidate in range(pool_size)
    )
    labels = tuple(
        SRECandidateLabel(
            candidate_id=candidate.candidate_id,
            relevant=position < positive_count,
            hard_negative=position >= positive_count,
            adversary=None if position < positive_count else (
                "near_duplicate" if position % 2 else "stale_ownership"
            ),
        )
        for position, candidate in enumerate(candidates)
    )
    return SRERetrievalCase(
        case_id=f"case-{index}",
        request_time="2026-02-01T00:00:00Z",
        query_text=f"find incident {index}",
        incoming_text=f"incident {index} is active",
        candidates=candidates,
        relationships=(
            SRERelationship(
                source_memory_id=candidates[0].candidate_id,
                destination_memory_id=candidates[1].candidate_id,
                edge_type="similar_to",
                effective_at="2026-01-01T00:00:00Z",
            ),
        ),
        labels=labels,
        scenario_family="family-a" if index % 2 else "family-b",
        entity_lineage=f"entity-{index}",
    )


def test_sealed_sre_paths_are_rejected_before_read(tmp_path: Path) -> None:
    inputs = tmp_path / "test.inputs.jsonl"
    labels = tmp_path / "labels.jsonl"
    inputs.write_text("")
    labels.write_text("")
    with pytest.raises(PermissionError, match="sealed SRE data"):
        load_sre_cases(inputs, labels)


def test_sre_development_split_is_deterministic_and_stratified() -> None:
    cases = tuple(
        _case(index, positive_count=index % 3)
        for index in range(20)
    )
    first = split_sre_development(cases, selection_count=10, seed=8)
    second = split_sre_development(tuple(reversed(cases)), selection_count=10, seed=8)
    assert [case.case_id for case in first[0]] == [case.case_id for case in second[0]]
    assert len(first[0]) == len(first[1]) == 10
    assert {case.case_id for case in first[0]}.isdisjoint(
        case.case_id for case in first[1]
    )


def test_runtime_features_are_finite_and_pool_aligned() -> None:
    case = _case(1)
    vocabulary = build_runtime_vocabulary((case,))
    features = build_candidate_features(case, vocabulary)
    assert features.shape == (case.pool_size, vocabulary.feature_width)
    assert np.isfinite(features).all()
    assert not np.array_equal(features[0], features[1])


def test_packed_sre_batch_enumerates_candidates_through_frontier() -> None:
    case = _case(2)
    vocabulary = build_runtime_vocabulary((case,))
    width = 8
    generator = torch.Generator().manual_seed(4)
    batch = pack_sre_batch(
        (case,),
        query_embeddings=torch.randn((1, width), generator=generator),
        incoming_embeddings=torch.randn((1, width), generator=generator),
        incoming_present=torch.tensor([True]),
        candidate_embeddings=torch.randn(
            (1, case.pool_size, width), generator=generator
        ),
        candidate_features=torch.from_numpy(
            build_candidate_features(case, vocabulary)
        ).unsqueeze(0),
        vocabulary=vocabulary,
    )
    expansion = batch.graph.topology.expand_frontier(batch.root_node_ids)
    assert expansion.total_arcs == case.pool_size
    assert torch.equal(expansion.destination_node_ids, batch.candidate_node_ids[0])
    neighbor_expansion = batch.graph.topology.expand_frontier(
        batch.candidate_node_ids.flatten()
    )
    assert neighbor_expansion.total_arcs == 2
    assert batch.graph.summaries.gather(expansion.destination_node_ids).total_rows == 4


def test_retrieval_metrics_reward_perfect_ranking_and_null_selection() -> None:
    scores = np.asarray([[4.0, 3.0, 1.0, 0.0], [0.0, 1.0, 3.0, 2.0]])
    relevance = np.asarray(
        [[True, True, False, False], [False, False, True, True]]
    )
    adversary = np.asarray([[-1, -1, 0, 1], [0, 1, -1, -1]], dtype=np.int16)
    metrics = evaluate_retrieval(
        scores=scores,
        relevance=relevance,
        adversary=adversary,
        tie_break=np.tile(np.arange(4), (2, 1)),
        null_scores=np.asarray([2.0, 1.5]),
        recall_k=2,
    )
    assert metrics["score"] == pytest.approx(1.0)
    assert metrics["set_selection"]["exact_set_accuracy"] == 1.0


def test_loader_keeps_supervision_out_of_runtime_schema(tmp_path: Path) -> None:
    case = _case(7)
    runtime = {
        "format": "sre-memory-runtime-input-v1",
        "case_id": case.case_id,
        "request_time": case.request_time,
        "query_text": case.query_text,
        "incoming_text": case.incoming_text,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "text": candidate.text,
                "memory_type": candidate.memory_type,
                "occurred_at": candidate.occurred_at,
                "status": candidate.status,
                "region": candidate.region,
            }
            for candidate in case.candidates
        ],
        "graph_edges": [
            {
                "source_memory_id": edge.source_memory_id,
                "destination_memory_id": edge.destination_memory_id,
                "edge_type": edge.edge_type,
                "effective_at": edge.effective_at,
            }
            for edge in case.relationships
        ],
    }
    labels = {
        "case_id": case.case_id,
        "positive_candidate_ids": list(case.positive_candidate_ids),
        "candidate_labels": [
            {
                "candidate_id": label.candidate_id,
                "relevant": label.relevant,
                "hard_negative": label.hard_negative,
                "adversary": "positive" if label.relevant else label.adversary,
            }
            for label in case.labels
        ],
        "groups": {
            "scenario_family": case.scenario_family,
            "entity_lineage": case.entity_lineage,
        },
    }
    inputs_path = tmp_path / "development.inputs.jsonl"
    labels_path = tmp_path / "development.labels.jsonl"
    inputs_path.write_text(json.dumps(runtime) + "\n")
    labels_path.write_text(json.dumps(labels) + "\n")
    loaded = load_sre_cases(inputs_path, labels_path, expected_pool_size=4)
    assert loaded[0].positive_candidate_ids == case.positive_candidate_ids
    assert "relevant" not in runtime["candidates"][0]
