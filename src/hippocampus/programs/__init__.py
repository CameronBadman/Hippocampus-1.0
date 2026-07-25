"""Exact synthetic graph programs and exchangeable manifold rendering."""

from .batching import PackedProgramBatch, pack_rendered_cases
from .counterfactuals import make_counterfactual
from .equivalent_views import make_equivalent_view
from .generator import GeneratorConfig, GraphProgramGenerator
from .leakage import MetadataLeakageReport, metadata_leakage_report
from .oracle import VerificationReport, verify_case
from .renderer import RenderedCase, SyntheticManifoldRenderer
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
from .splits import SplitManifest, SplitSpec, build_split_manifest, default_split_specs

__all__ = [
    "CandidateTarget",
    "CounterfactualKind",
    "GeneratorConfig",
    "GraphProgramCase",
    "GraphProgramGenerator",
    "Intervention",
    "MetadataLeakageReport",
    "ObservableAtom",
    "OracleRound",
    "PackedProgramBatch",
    "ParallelOracleTrace",
    "ProgramEdge",
    "ProgramFamily",
    "ProgramNode",
    "RenderedCase",
    "SplitManifest",
    "SplitSpec",
    "SyntheticManifoldRenderer",
    "TerminationDecision",
    "TerminationTarget",
    "VerificationReport",
    "build_split_manifest",
    "default_split_specs",
    "make_counterfactual",
    "make_equivalent_view",
    "metadata_leakage_report",
    "pack_rendered_cases",
    "verify_case",
]
