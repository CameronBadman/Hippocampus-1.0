from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .encoding import (
    SREEncodedCases,
    case_runtime_fingerprint,
    load_encoded_cases,
)
from .features import SREVocabulary, build_runtime_vocabulary, stable_u63
from .metrics import evaluate_retrieval
from .model import PackedSRECanonicalRetriever
from .packed import pack_sre_batch
from .schema import SRERetrievalCase, load_sre_cases, split_sre_development


DEFAULT_SOURCE_ROOT = Path(
    "/home/cameron/projects/hippo-qwen-2/"
    "autoresearch/sre_incident_memory_v3/locked"
)
DEFAULT_DEMO_ROOT = Path("/home/cameron/projects/hippo-qwen-2/demo_data/sre")


@dataclass(frozen=True, slots=True)
class SREPartition:
    cases: tuple[SRERetrievalCase, ...]
    encoded: SREEncodedCases

    def __post_init__(self) -> None:
        self.encoded.validate(self.cases)

    def select(self, indices: Sequence[int] | torch.Tensor) -> "SREPartition":
        values = torch.as_tensor(indices, dtype=torch.int64)
        cases = tuple(self.cases[int(index)] for index in values.tolist())
        encoded = replace(
            self.encoded.select(values),
            runtime_fingerprint=case_runtime_fingerprint(cases),
        )
        return SREPartition(
            cases=cases,
            encoded=encoded,
        )


@dataclass(frozen=True, slots=True)
class SREDevelopmentCorpus:
    train: SREPartition
    selection: SREPartition
    evaluation: SREPartition
    demo: SREPartition
    vocabulary: SREVocabulary


def load_development_corpus(
    *,
    cache_root: str | Path,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    demo_root: str | Path = DEFAULT_DEMO_ROOT,
) -> SREDevelopmentCorpus:
    source = Path(source_root)
    demo_source = Path(demo_root)
    train = load_sre_cases(
        source / "train.inputs.jsonl",
        source / "train.labels.jsonl",
    )
    validation = load_sre_cases(
        source / "validation.inputs.jsonl",
        source / "validation.labels.jsonl",
    )
    selection, evaluation = split_sre_development(
        validation,
        selection_count=100,
    )
    demo = load_sre_cases(
        demo_source / "demo.inputs.jsonl",
        demo_source / "demo.labels.jsonl",
    )
    cache = Path(cache_root)
    vocabulary = build_runtime_vocabulary(train)
    return SREDevelopmentCorpus(
        train=SREPartition(
            train,
            load_encoded_cases(cache / "train.npz", cases=train),
        ),
        selection=SREPartition(
            selection,
            load_encoded_cases(cache / "selection.npz", cases=selection),
        ),
        evaluation=SREPartition(
            evaluation,
            load_encoded_cases(cache / "evaluation.npz", cases=evaluation),
        ),
        demo=SREPartition(
            demo,
            load_encoded_cases(cache / "demo.npz", cases=demo),
        ),
        vocabulary=vocabulary,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def pack_partition(
    partition: SREPartition,
    *,
    vocabulary: SREVocabulary,
    device: torch.device | str,
    execution_policy: str | None = None,
    validate: bool = False,
):
    encoded = partition.encoded
    return pack_sre_batch(
        partition.cases,
        query_embeddings=encoded.query_embeddings,
        incoming_embeddings=encoded.incoming_embeddings,
        incoming_present=encoded.incoming_present,
        candidate_embeddings=encoded.candidate_embeddings,
        candidate_features=encoded.candidate_features,
        vocabulary=vocabulary,
        device=device,
        execution_policy=execution_policy,
        validate=validate,
    )


def permute_partition_candidates(
    partition: SREPartition,
    permutations: torch.Tensor,
) -> SREPartition:
    """Apply an aligned candidate insertion-order permutation to each case."""

    permutations = torch.as_tensor(permutations, dtype=torch.int64, device="cpu")
    expected = (len(partition.cases), partition.encoded.pool_size)
    if tuple(permutations.shape) != expected:
        raise ValueError(f"permutations must have shape {expected}")
    identity = torch.arange(partition.encoded.pool_size)
    if any(
        not torch.equal(torch.sort(row).values, identity)
        for row in permutations
    ):
        raise ValueError("each candidate mapping must be a permutation")
    cases = tuple(
        SRERetrievalCase(
            case_id=case.case_id,
            request_time=case.request_time,
            query_text=case.query_text,
            incoming_text=case.incoming_text,
            candidates=tuple(case.candidates[int(index)] for index in order),
            relationships=case.relationships,
            labels=tuple(case.labels[int(index)] for index in order),
            scenario_family=case.scenario_family,
            entity_lineage=case.entity_lineage,
        )
        for case, order in zip(partition.cases, permutations.tolist(), strict=True)
    )
    gather = permutations[:, :, None].expand(
        -1,
        -1,
        partition.encoded.width,
    )
    feature_gather = permutations[:, :, None].expand(
        -1,
        -1,
        partition.encoded.candidate_features.shape[-1],
    )
    encoded = SREEncodedCases(
        case_ids=partition.encoded.case_ids,
        query_embeddings=partition.encoded.query_embeddings,
        incoming_embeddings=partition.encoded.incoming_embeddings,
        incoming_present=partition.encoded.incoming_present,
        candidate_embeddings=torch.gather(
            partition.encoded.candidate_embeddings,
            1,
            gather,
        ),
        candidate_features=torch.gather(
            partition.encoded.candidate_features,
            1,
            feature_gather,
        ),
        encoder_name=partition.encoded.encoder_name,
        encoder_revision=partition.encoded.encoder_revision,
        runtime_fingerprint=case_runtime_fingerprint(cases),
    )
    return SREPartition(cases=cases, encoded=encoded)


def _matrix_labels(partition: SREPartition) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relevance = np.asarray([case.relevance for case in partition.cases], dtype=bool)
    family_names = (
        "query_echo",
        "answer_shaped",
        "stale_ownership",
        "rolled_back_deployment",
        "superseded_fix",
        "contradictory_memory",
        "same_entity_wrong_time",
        "same_entity_wrong_relationship",
        "near_duplicate",
        "relationship_path_decoy",
        "mixed_adversarial",
    )
    family_ids = {name: index for index, name in enumerate(family_names)}
    adversary = np.asarray(
        [
            [
                -1 if label.relevant else family_ids[label.adversary or "mixed_adversarial"]
                for label in case.labels
            ]
            for case in partition.cases
        ],
        dtype=np.int16,
    )
    tie_break = np.asarray(
        [
            [
                stable_u63(candidate.candidate_id)
                for candidate in case.candidates
            ]
            for case in partition.cases
        ],
        dtype=np.int64,
    )
    return relevance, adversary, tie_break


def retrieval_metrics(
    partition: SREPartition,
    *,
    scores: np.ndarray,
    null_scores: np.ndarray,
) -> dict[str, Any]:
    relevance, adversary, tie_break = _matrix_labels(partition)
    metrics = evaluate_retrieval(
        scores=scores,
        relevance=relevance,
        adversary=adversary,
        tie_break=tie_break,
        null_scores=null_scores,
        scenario_families=[case.scenario_family for case in partition.cases],
    )
    metrics["enumerated_candidate_coverage"] = 1.0
    metrics["arcs_scored"] = len(partition.cases) * partition.cases[0].pool_size
    return metrics


def evaluate_model(
    model: PackedSRECanonicalRetriever,
    partition: SREPartition,
    *,
    vocabulary: SREVocabulary,
    device: torch.device | str,
    batch_size: int = 16,
    deterministic: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    null_scores = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(partition.cases), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(partition.cases)))
            packed = pack_partition(
                partition.select(indices),
                vocabulary=vocabulary,
                device=device,
                execution_policy="deterministic" if deterministic else None,
            )
            output = model(packed)
            scores.append(output.scores.float().cpu())
            null_scores.append(output.null_scores.float().cpu())
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - started
    score_values = torch.cat(scores).numpy()
    null_values = torch.cat(null_scores).numpy()
    metrics = retrieval_metrics(
        partition,
        scores=score_values,
        null_scores=null_values,
    )
    metrics["latency_seconds"] = elapsed
    metrics["latency_ms_per_case"] = 1_000 * elapsed / len(partition.cases)
    return metrics, score_values, null_values


def model_order_invariance(
    model: PackedSRECanonicalRetriever,
    partition: SREPartition,
    *,
    vocabulary: SREVocabulary,
    device: torch.device | str,
    seed: int = 80_008,
    case_count: int = 8,
) -> dict[str, Any]:
    selected = partition.select(torch.arange(min(case_count, len(partition.cases))))
    generator = torch.Generator().manual_seed(seed)
    permutations = torch.stack(
        [
            torch.randperm(selected.encoded.pool_size, generator=generator)
            for _ in selected.cases
        ]
    )
    permuted = permute_partition_candidates(selected, permutations)
    model.eval()
    execution_policy = (
        "deterministic"
        if torch.are_deterministic_algorithms_enabled()
        else None
    )
    with torch.inference_mode():
        original = model(
            pack_partition(
                selected,
                vocabulary=vocabulary,
                device=device,
                execution_policy=execution_policy,
            )
        )
        changed = model(
            pack_partition(
                permuted,
                vocabulary=vocabulary,
                device=device,
                execution_policy=execution_policy,
            )
        )
    restored = torch.empty_like(changed.scores)
    restored.scatter_(1, permutations.to(changed.scores.device), changed.scores)
    score_delta = torch.abs(original.scores - restored)
    original_decisions = original.scores > original.null_scores[:, None]
    changed_decisions = restored > changed.null_scores[:, None]
    return {
        "maximum_score_delta": float(score_delta.max().item()),
        "decision_mismatch_count": int(
            (original_decisions != changed_decisions).sum().item()
        ),
        "null_score_delta": float(
            torch.abs(original.null_scores - changed.null_scores).max().item()
        ),
        "case_count": len(selected.cases),
    }


def frozen_baseline_scores(
    partition: SREPartition,
) -> tuple[np.ndarray, np.ndarray]:
    encoded = partition.encoded
    query = F.normalize(encoded.query_embeddings.float(), dim=-1)
    incoming = F.normalize(encoded.incoming_embeddings.float(), dim=-1)
    candidates = F.normalize(encoded.candidate_embeddings.float(), dim=-1)
    query_similarity = torch.einsum("bd,bkd->bk", query, candidates)
    incoming_similarity = torch.einsum("bd,bkd->bk", incoming, candidates)
    incoming_similarity = incoming_similarity * encoded.incoming_present[:, None]
    features = encoded.candidate_features.float()
    scores = (
        2.0 * query_similarity
        + 0.8 * incoming_similarity
        + 0.5 * features[:, :, 0]
        + 0.3 * features[:, :, 2]
        + 0.6 * features[:, :, 4]
        + 0.2 * features[:, :, 5]
        + 0.3 * features[:, :, 13]
        - 0.5 * features[:, :, 14]
        - 0.4 * features[:, :, 15]
        - 0.8 * features[:, :, 16]
    )
    null = torch.full((len(partition.cases),), 1.0)
    return scores.numpy(), null.numpy()


def evaluate_frozen_baseline(
    partition: SREPartition,
    *,
    vocabulary: SREVocabulary,
    device: torch.device | str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    # Validate that the comparison sees exactly the packed CSR candidate set.
    packed = pack_partition(
        partition.select(torch.arange(min(8, len(partition.cases)))),
        vocabulary=vocabulary,
        device=device,
        validate=True,
    )
    packed.validate_execution_path()
    scores, null_scores = frozen_baseline_scores(partition)
    return (
        retrieval_metrics(
            partition,
            scores=scores,
            null_scores=null_scores,
        ),
        scores,
        null_scores,
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: PackedSRECanonicalRetriever,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    seed: int,
    selection_metrics: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "spider-v0.8-sre-checkpoint-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "seed": seed,
            "selection_metrics": selection_metrics,
        },
        path,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
