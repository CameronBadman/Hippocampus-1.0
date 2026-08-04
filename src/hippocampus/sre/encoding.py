from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .features import SREVocabulary, build_candidate_features
from .schema import SRERetrievalCase


DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class TextEncoder(Protocol):
    @property
    def width(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class FrozenMiniLMEncoder:
    """Frozen, position-agnostic sentence encoder used only at ingestion."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_ENCODER,
        revision: str = DEFAULT_ENCODER_REVISION,
        device: torch.device | str = "cpu",
        batch_size: int = 256,
        max_length: int = 160,
        local_files_only: bool = True,
        model_path: str | Path | None = None,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("encoder batch size and maximum length must be positive")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "SRE encoding requires the optional 'sre' dependencies"
            ) from error
        reference = str(model_path) if model_path is not None else model_name
        kwargs = {"local_files_only": local_files_only}
        if model_path is None:
            kwargs["revision"] = revision
        self._tokenizer = AutoTokenizer.from_pretrained(reference, **kwargs)
        self._model = AutoModel.from_pretrained(reference, **kwargs)
        self._device = torch.device(device)
        self._model.to(self._device).eval()
        self._batch_size = batch_size
        self._max_length = max_length
        self.model_name = model_name
        self.revision = revision

    @property
    def width(self) -> int:
        return int(self._model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.width), dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self._batch_size):
                rows = list(texts[start : start + self._batch_size])
                tokens = self._tokenizer(
                    rows,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                tokens = {name: value.to(self._device) for name, value in tokens.items()}
                hidden = self._model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                outputs.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)


def case_runtime_fingerprint(cases: Sequence[SRERetrievalCase]) -> str:
    """Hash all model-visible observations, never supervisor labels."""

    digest = hashlib.sha256()
    for case in cases:
        visible = {
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
            "relationships": [
                {
                    "source": edge.source_memory_id,
                    "destination": edge.destination_memory_id,
                    "edge_type": edge.edge_type,
                    "effective_at": edge.effective_at,
                }
                for edge in case.relationships
            ],
        }
        digest.update(json.dumps(visible, sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SREEncodedCases:
    case_ids: tuple[str, ...]
    query_embeddings: torch.Tensor
    incoming_embeddings: torch.Tensor
    incoming_present: torch.Tensor
    candidate_embeddings: torch.Tensor
    candidate_features: torch.Tensor
    encoder_name: str
    encoder_revision: str
    runtime_fingerprint: str

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    @property
    def pool_size(self) -> int:
        return int(self.candidate_embeddings.shape[1])

    @property
    def width(self) -> int:
        return int(self.query_embeddings.shape[1])

    def validate(
        self,
        cases: Sequence[SRERetrievalCase] | None = None,
    ) -> "SREEncodedCases":
        count = self.case_count
        if len(set(self.case_ids)) != count:
            raise ValueError("encoded SRE case IDs are not unique")
        if self.query_embeddings.shape != self.incoming_embeddings.shape:
            raise ValueError("query and incoming embeddings are not aligned")
        if tuple(self.query_embeddings.shape) != (count, self.width):
            raise ValueError("query embeddings have an invalid shape")
        if tuple(self.incoming_present.shape) != (count,):
            raise ValueError("incoming presence is not case-aligned")
        if tuple(self.candidate_embeddings.shape[:2]) != (
            count,
            self.pool_size,
        ):
            raise ValueError("candidate embeddings are not case-aligned")
        if self.candidate_embeddings.shape[2] != self.width:
            raise ValueError("candidate and query embedding widths disagree")
        if tuple(self.candidate_features.shape[:2]) != (
            count,
            self.pool_size,
        ):
            raise ValueError("candidate features are not case-aligned")
        tensors = (
            self.query_embeddings,
            self.incoming_embeddings,
            self.candidate_embeddings,
            self.candidate_features,
        )
        if any(not value.is_floating_point() for value in tensors):
            raise TypeError("encoded SRE values must be floating point")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("encoded SRE values contain non-finite entries")
        if self.incoming_present.dtype != torch.bool:
            raise TypeError("incoming_present must be boolean")
        if cases is not None:
            expected_ids = tuple(case.case_id for case in cases)
            if expected_ids != self.case_ids:
                raise ValueError("encoded cache case order disagrees with the corpus")
            if case_runtime_fingerprint(cases) != self.runtime_fingerprint:
                raise ValueError("encoded cache runtime fingerprint disagrees")
        return self

    def select(self, indices: Sequence[int] | torch.Tensor) -> "SREEncodedCases":
        ids = torch.as_tensor(indices, dtype=torch.int64)
        selected = tuple(self.case_ids[int(index)] for index in ids.tolist())
        return SREEncodedCases(
            case_ids=selected,
            query_embeddings=self.query_embeddings[ids],
            incoming_embeddings=self.incoming_embeddings[ids],
            incoming_present=self.incoming_present[ids],
            candidate_embeddings=self.candidate_embeddings[ids],
            candidate_features=self.candidate_features[ids],
            encoder_name=self.encoder_name,
            encoder_revision=self.encoder_revision,
            runtime_fingerprint=self.runtime_fingerprint,
        )


def encode_sre_cases(
    cases: Sequence[SRERetrievalCase],
    *,
    vocabulary: SREVocabulary,
    encoder: TextEncoder,
    encoder_name: str = DEFAULT_ENCODER,
    encoder_revision: str = DEFAULT_ENCODER_REVISION,
) -> SREEncodedCases:
    if not cases:
        raise ValueError("cannot encode an empty SRE corpus")
    pool_size = cases[0].pool_size
    if any(case.pool_size != pool_size for case in cases):
        raise ValueError("encoded SRE pools must have a common size")
    query = encoder.encode([case.query_text for case in cases]).float()
    incoming_present = torch.tensor(
        [case.incoming_text is not None for case in cases], dtype=torch.bool
    )
    incoming = encoder.encode(
        [case.incoming_text or case.query_text for case in cases]
    ).float()
    candidate = encoder.encode(
        [candidate.text for case in cases for candidate in case.candidates]
    ).float().reshape(len(cases), pool_size, encoder.width)
    features = torch.from_numpy(
        np.stack(
            [build_candidate_features(case, vocabulary) for case in cases]
        )
    ).float()
    return SREEncodedCases(
        case_ids=tuple(case.case_id for case in cases),
        query_embeddings=query,
        incoming_embeddings=incoming,
        incoming_present=incoming_present,
        candidate_embeddings=candidate,
        candidate_features=features,
        encoder_name=encoder_name,
        encoder_revision=encoder_revision,
        runtime_fingerprint=case_runtime_fingerprint(cases),
    ).validate(cases)


def save_encoded_cases(path: str | Path, encoded: SREEncodedCases) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.validate()
    np.savez(
        path,
        case_ids=np.asarray(encoded.case_ids),
        query_embeddings=encoded.query_embeddings.numpy().astype(np.float16),
        incoming_embeddings=encoded.incoming_embeddings.numpy().astype(np.float16),
        incoming_present=encoded.incoming_present.numpy(),
        candidate_embeddings=encoded.candidate_embeddings.numpy().astype(np.float16),
        candidate_features=encoded.candidate_features.numpy().astype(np.float16),
        encoder_name=np.asarray(encoded.encoder_name),
        encoder_revision=np.asarray(encoded.encoder_revision),
        runtime_fingerprint=np.asarray(encoded.runtime_fingerprint),
    )


def load_encoded_cases(
    path: str | Path,
    *,
    cases: Sequence[SRERetrievalCase] | None = None,
) -> SREEncodedCases:
    with np.load(Path(path), allow_pickle=False) as values:
        encoded = SREEncodedCases(
            case_ids=tuple(str(value) for value in values["case_ids"].tolist()),
            query_embeddings=torch.from_numpy(values["query_embeddings"].astype(np.float32)),
            incoming_embeddings=torch.from_numpy(
                values["incoming_embeddings"].astype(np.float32)
            ),
            incoming_present=torch.from_numpy(values["incoming_present"].astype(bool)),
            candidate_embeddings=torch.from_numpy(
                values["candidate_embeddings"].astype(np.float32)
            ),
            candidate_features=torch.from_numpy(
                values["candidate_features"].astype(np.float32)
            ),
            encoder_name=str(values["encoder_name"].item()),
            encoder_revision=str(values["encoder_revision"].item()),
            runtime_fingerprint=str(values["runtime_fingerprint"].item()),
        )
    return encoded.validate(cases)
