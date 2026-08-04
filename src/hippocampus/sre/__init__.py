"""Packed-graph transfer utilities for synthetic SRE memory retrieval."""

from .features import (
    SREVocabulary,
    build_candidate_features,
    build_runtime_vocabulary,
    stable_u63,
)
from .experiment import (
    SREDevelopmentCorpus,
    SREPartition,
    evaluate_frozen_baseline,
    evaluate_model,
    frozen_baseline_scores,
    load_development_corpus,
    model_order_invariance,
    pack_partition,
    permute_partition_candidates,
    retrieval_metrics,
    seed_everything,
)
from .encoding import (
    DEFAULT_ENCODER,
    DEFAULT_ENCODER_REVISION,
    FrozenMiniLMEncoder,
    SREEncodedCases,
    case_runtime_fingerprint,
    encode_sre_cases,
    load_encoded_cases,
    save_encoded_cases,
)
from .metrics import evaluate_retrieval
from .losses import SRERetrievalLossConfig, sre_retrieval_loss
from .model import (
    PackedSRECanonicalRetriever,
    SREModelConfig,
    SRERetrievalOutput,
)
from .packed import PackedSRERetrievalBatch, pack_sre_batch
from .schema import (
    SRE_ADVERSARY_FAMILIES,
    SRECandidate,
    SRECandidateLabel,
    SRERelationship,
    SRERetrievalCase,
    load_sre_cases,
    split_sre_development,
)

__all__ = [
    "PackedSRERetrievalBatch",
    "PackedSRECanonicalRetriever",
    "DEFAULT_ENCODER",
    "DEFAULT_ENCODER_REVISION",
    "FrozenMiniLMEncoder",
    "SRECandidate",
    "SRECandidateLabel",
    "SRERelationship",
    "SRERetrievalCase",
    "SREVocabulary",
    "SREEncodedCases",
    "SREDevelopmentCorpus",
    "SREModelConfig",
    "SRERetrievalLossConfig",
    "SRERetrievalOutput",
    "SREPartition",
    "SRE_ADVERSARY_FAMILIES",
    "build_candidate_features",
    "build_runtime_vocabulary",
    "case_runtime_fingerprint",
    "encode_sre_cases",
    "evaluate_retrieval",
    "evaluate_frozen_baseline",
    "evaluate_model",
    "frozen_baseline_scores",
    "load_sre_cases",
    "load_encoded_cases",
    "load_development_corpus",
    "model_order_invariance",
    "pack_sre_batch",
    "pack_partition",
    "permute_partition_candidates",
    "retrieval_metrics",
    "split_sre_development",
    "save_encoded_cases",
    "seed_everything",
    "sre_retrieval_loss",
    "stable_u63",
]
