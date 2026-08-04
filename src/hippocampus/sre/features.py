from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import re
from typing import Sequence

import numpy as np

from .schema import SRERetrievalCase


_WORD_RE = re.compile(r"[a-z0-9]+(?:[-_.:/][a-z0-9]+)*")
_INCIDENT_RE = re.compile(r"\binc-[a-z0-9-]+\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SREVocabulary:
    memory_types: tuple[str, ...]
    statuses: tuple[str, ...]
    regions: tuple[str, ...]
    edge_types: tuple[str, ...]

    @staticmethod
    def _values(values: set[str]) -> tuple[str, ...]:
        return ("<unknown>", *sorted(values))

    @property
    def feature_width(self) -> int:
        return 19 + len(self.memory_types) + len(self.statuses) + len(
            self.regions
        ) + 2 * len(self.edge_types)


def build_runtime_vocabulary(
    cases: Sequence[SRERetrievalCase],
) -> SREVocabulary:
    if not cases:
        raise ValueError("cannot build an SRE vocabulary from no cases")
    return SREVocabulary(
        memory_types=SREVocabulary._values(
            {candidate.memory_type for case in cases for candidate in case.candidates}
        ),
        statuses=SREVocabulary._values(
            {candidate.status for case in cases for candidate in case.candidates}
        ),
        regions=SREVocabulary._values(
            {candidate.region for case in cases for candidate in case.candidates}
        ),
        edge_types=SREVocabulary._values(
            {edge.edge_type for case in cases for edge in case.relationships}
        ),
    )


def stable_u63(value: str) -> int:
    digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _containment(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _lookup(values: tuple[str, ...], value: str) -> int:
    try:
        return values.index(value)
    except ValueError:
        return 0


def build_candidate_features(
    case: SRERetrievalCase,
    vocabulary: SREVocabulary,
) -> np.ndarray:
    """Build runtime-visible numeric and categorical candidate features."""

    candidate_index = {
        candidate.candidate_id: index
        for index, candidate in enumerate(case.candidates)
    }
    incoming_types = [Counter() for _ in case.candidates]
    outgoing_types = [Counter() for _ in case.candidates]
    for edge in case.relationships:
        source = candidate_index[edge.source_memory_id]
        destination = candidate_index[edge.destination_memory_id]
        outgoing_types[source][edge.edge_type] += 1
        incoming_types[destination][edge.edge_type] += 1
    query_tokens = _tokens(case.query_text)
    incoming_tokens = _tokens(case.incoming_text or "")
    query_incidents = set(_INCIDENT_RE.findall(case.query_text.lower()))
    request_time = _timestamp(case.request_time)
    rows: list[np.ndarray] = []
    for index, candidate in enumerate(case.candidates):
        candidate_tokens = _tokens(candidate.text)
        candidate_incidents = set(_INCIDENT_RE.findall(candidate.text.lower()))
        age_seconds = (request_time - _timestamp(candidate.occurred_at)).total_seconds()
        age_days = age_seconds / 86_400.0
        status = candidate.status.lower()
        numeric = np.asarray(
            [
                _jaccard(query_tokens, candidate_tokens),
                _jaccard(incoming_tokens, candidate_tokens),
                _containment(query_tokens, candidate_tokens),
                _containment(incoming_tokens, candidate_tokens),
                _jaccard(query_incidents, candidate_incidents),
                float(candidate.region.lower() in f"{case.query_text} {case.incoming_text or ''}".lower()),
                math.copysign(math.log1p(abs(age_days)), age_days),
                math.log1p(abs(age_days)),
                float(age_seconds >= 0),
                float(age_seconds < 0),
                float(0 <= age_seconds <= 3_600),
                float(0 <= age_seconds <= 86_400),
                float(0 <= age_seconds <= 7 * 86_400),
                float("active" in status or status in {"confirmed", "successful", "validated"}),
                float("supersed" in status or "stale" in status),
                float(any(word in status for word in ("failed", "rejected", "retracted"))),
                float("unverified" in status),
                min(sum(incoming_types[index].values()), 8) / 8.0,
                min(sum(outgoing_types[index].values()), 8) / 8.0,
            ],
            dtype=np.float32,
        )
        categorical = np.zeros(vocabulary.feature_width - 19, dtype=np.float32)
        offset = 0
        categorical[offset + _lookup(vocabulary.memory_types, candidate.memory_type)] = 1
        offset += len(vocabulary.memory_types)
        categorical[offset + _lookup(vocabulary.statuses, candidate.status)] = 1
        offset += len(vocabulary.statuses)
        categorical[offset + _lookup(vocabulary.regions, candidate.region)] = 1
        offset += len(vocabulary.regions)
        for edge_type, count in incoming_types[index].items():
            categorical[offset + _lookup(vocabulary.edge_types, edge_type)] = min(count, 4) / 4
        offset += len(vocabulary.edge_types)
        for edge_type, count in outgoing_types[index].items():
            categorical[offset + _lookup(vocabulary.edge_types, edge_type)] = min(count, 4) / 4
        rows.append(np.concatenate((numeric, categorical)))
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (case.pool_size, vocabulary.feature_width):
        raise AssertionError("SRE feature shape drifted")
    if not np.isfinite(result).all():
        raise ValueError("SRE features contain non-finite values")
    return result


def bm25_scores(query: str, documents: Sequence[str]) -> np.ndarray:
    query_terms = list(_WORD_RE.findall(query.lower()))
    document_terms = [list(_WORD_RE.findall(document.lower())) for document in documents]
    if not query_terms:
        return np.zeros(len(documents), dtype=np.float32)
    frequency = Counter()
    for terms in document_terms:
        frequency.update(set(terms))
    average_length = sum(map(len, document_terms)) / max(len(documents), 1)
    result = []
    for terms in document_terms:
        counts = Counter(terms)
        norm = 1.5 * (0.25 + 0.75 * len(terms) / max(average_length, 1.0))
        score = 0.0
        for term in query_terms:
            document_frequency = frequency[term]
            inverse = math.log(
                1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            count = counts[term]
            score += inverse * (count * 2.5) / (count + norm)
        result.append(score)
    values = np.asarray(result, dtype=np.float32)
    low = float(values.min(initial=0))
    high = float(values.max(initial=0))
    return np.zeros_like(values) if high - low < 1e-8 else (values - low) / (high - low)
