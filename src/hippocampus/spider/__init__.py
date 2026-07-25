"""Trainable recurrent manifold graph interpreter and deterministic controller."""

from .baselines import FlatTransformerScorer, PooledScorer
from .config import SparseControllerConfig, SpiderModelConfig
from .controller import (
    ControllerResult,
    ControllerState,
    ControllerStep,
    EvidenceLedgerEntry,
    SparseWavefrontController,
    TraceLedgerEntry,
    stable_candidate_selection,
)
from .hypothesis import HypothesisBatch
from .model import CandidateScorerBase, SpiderModel
from .set_attention import (
    AttentionBackendStatus,
    PositionFreeCrossAttention,
    attention_backend_status,
    padded_family_gather,
)
from .types import CandidateOutputs, PaddedSet

__all__ = [
    "AttentionBackendStatus",
    "CandidateOutputs",
    "CandidateScorerBase",
    "ControllerResult",
    "ControllerState",
    "ControllerStep",
    "EvidenceLedgerEntry",
    "FlatTransformerScorer",
    "HypothesisBatch",
    "PaddedSet",
    "PooledScorer",
    "PositionFreeCrossAttention",
    "SparseControllerConfig",
    "SparseWavefrontController",
    "SpiderModel",
    "SpiderModelConfig",
    "TraceLedgerEntry",
    "attention_backend_status",
    "padded_family_gather",
    "stable_candidate_selection",
]
