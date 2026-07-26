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
from .experiment import (
    ResolvedExperiment,
    build_model,
    load_experiment,
    parameter_count,
)
from .evaluation import EvaluationReport, evaluate_batches
from .losses import (
    CandidateSupervision,
    LossTerm,
    SpiderLossConfig,
    SpiderLossReport,
    candidate_loss_report,
    multi_positive_priority_loss,
)
from .model import CandidateScorerBase, SpiderModel
from .set_attention import (
    AttentionBackendStatus,
    PositionFreeCrossAttention,
    attention_backend_status,
    padded_family_gather,
)
from .types import CandidateOutputs, PaddedSet
from .training import (
    OracleMetrics,
    OracleRollout,
    TrainingLoopConfig,
    TrainingRecord,
    TrainingResult,
    evaluate_oracle_batches,
    make_tiny_cases,
    oracle_rollout,
    train_oracle_batches,
)

__all__ = [
    "AttentionBackendStatus",
    "CandidateOutputs",
    "CandidateScorerBase",
    "CandidateSupervision",
    "ControllerResult",
    "ControllerState",
    "ControllerStep",
    "EvidenceLedgerEntry",
    "EvaluationReport",
    "FlatTransformerScorer",
    "HypothesisBatch",
    "LossTerm",
    "OracleMetrics",
    "OracleRollout",
    "PaddedSet",
    "PooledScorer",
    "PositionFreeCrossAttention",
    "ResolvedExperiment",
    "SparseControllerConfig",
    "SparseWavefrontController",
    "SpiderModel",
    "SpiderModelConfig",
    "SpiderLossConfig",
    "SpiderLossReport",
    "TraceLedgerEntry",
    "TrainingLoopConfig",
    "TrainingRecord",
    "TrainingResult",
    "attention_backend_status",
    "build_model",
    "candidate_loss_report",
    "evaluate_oracle_batches",
    "evaluate_batches",
    "make_tiny_cases",
    "multi_positive_priority_loss",
    "load_experiment",
    "oracle_rollout",
    "parameter_count",
    "padded_family_gather",
    "stable_candidate_selection",
    "train_oracle_batches",
]
