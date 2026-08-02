"""Exact synthetic graph programs and exchangeable manifold rendering."""

from .batching import PackedProgramBatch, pack_rendered_cases
from .identifiability import (
    IdentifiabilityProbeConfig,
    IdentifiabilityReport,
    PairIdentifiability,
    run_renderer_identifiability,
)
from .counterfactuals import make_counterfactual
from .equivalent_views import make_equivalent_view
from .generator import GeneratorConfig, GraphProgramGenerator
from .leakage import MetadataLeakageReport, metadata_leakage_report
from .oracle import VerificationReport, verify_case
from .renderer import (
    RenderedCase,
    RendererGeometry,
    SyntheticManifoldRenderer,
)
from .recurrence import (
    RECURRENCE_DATASET_VERSION,
    RecurrenceLeakageReport,
    RecurrenceNecessityManifest,
    RecurrenceNecessitySpec,
    build_recurrence_necessity_manifest,
    default_recurrence_necessity_specs,
    generate_recurrence_necessity_cases,
    generate_recurrence_necessity_pair,
    recurrence_metadata_leakage_report,
)
from .schema import (
    CandidateTarget,
    CounterfactualKind,
    GraphProgramCase,
    Intervention,
    ObservableAtom,
    OracleRound,
    ParallelOracleTrace,
    ProgramEdge,
    ProgramFamily,
    ProgramNode,
    TerminationDecision,
    TerminationTarget,
)
from .swapping import FunctionalEdgeSwap, swap_aligned_edge_manifolds
from .splits import (
    SplitManifest,
    SplitSpec,
    build_split_manifest,
    default_split_specs,
    default_split_specs_v0_2,
    generate_split_cases,
)
from .stress import (
    RolloutStressExample,
    RolloutStressKind,
    RolloutStressManifest,
    build_rollout_stress_manifest,
    generate_rollout_stress_examples,
)

__all__ = [
    "CandidateTarget",
    "CounterfactualKind",
    "GeneratorConfig",
    "GraphProgramCase",
    "GraphProgramGenerator",
    "FunctionalEdgeSwap",
    "Intervention",
    "MetadataLeakageReport",
    "ObservableAtom",
    "OracleRound",
    "PackedProgramBatch",
    "IdentifiabilityProbeConfig",
    "IdentifiabilityReport",
    "PairIdentifiability",
    "ParallelOracleTrace",
    "ProgramEdge",
    "ProgramFamily",
    "ProgramNode",
    "RECURRENCE_DATASET_VERSION",
    "RecurrenceLeakageReport",
    "RecurrenceNecessityManifest",
    "RecurrenceNecessitySpec",
    "RenderedCase",
    "RendererGeometry",
    "RolloutStressExample",
    "RolloutStressKind",
    "RolloutStressManifest",
    "SplitManifest",
    "SplitSpec",
    "SyntheticManifoldRenderer",
    "TerminationDecision",
    "TerminationTarget",
    "VerificationReport",
    "build_split_manifest",
    "build_rollout_stress_manifest",
    "build_recurrence_necessity_manifest",
    "default_split_specs",
    "default_split_specs_v0_2",
    "default_recurrence_necessity_specs",
    "generate_recurrence_necessity_cases",
    "generate_recurrence_necessity_pair",
    "generate_rollout_stress_examples",
    "generate_split_cases",
    "make_counterfactual",
    "make_equivalent_view",
    "metadata_leakage_report",
    "pack_rendered_cases",
    "run_renderer_identifiability",
    "recurrence_metadata_leakage_report",
    "swap_aligned_edge_manifolds",
    "verify_case",
]
