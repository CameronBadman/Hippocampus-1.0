from __future__ import annotations

import pytest
import torch

from hippocampus import GraphSchema
from hippocampus.programs import (
    IdentifiabilityProbeConfig,
    SyntheticManifoldRenderer,
    run_renderer_identifiability,
)


def _renderer(geometry: str) -> SyntheticManifoldRenderer:
    return SyntheticManifoldRenderer(
        GraphSchema(summary_dim=16, context_dim=16, edge_dim=16),
        query_dim=16,
        seed=41,
        geometry=geometry,
    )


def test_a0_remains_modality_independent() -> None:
    renderer = _renderer("independent")

    query = renderer.render_symbol("unseen_symbol", modality="query")
    edge = renderer.render_symbol("unseen_symbol", modality="edge")

    assert renderer.renderer_version == "renderer-v0.4"
    assert not torch.allclose(query, edge)


def test_a1_shares_symbol_displacements_across_modalities() -> None:
    renderer = _renderer("shared_additive")

    query_delta = (
        renderer.render_symbol("left", modality="query")
        - renderer.render_symbol("right", modality="query")
    )
    edge_delta = (
        renderer.render_symbol("left", modality="edge")
        - renderer.render_symbol("right", modality="edge")
    )

    assert torch.allclose(query_delta, edge_delta, atol=1e-6, rtol=0.0)


def test_a2_modality_transforms_are_seeded_and_orthogonal() -> None:
    first = _renderer("orthogonal_aligned")
    replay = _renderer("orthogonal_aligned")

    for modality in ("query", "summary", "context", "edge"):
        transform = first.modality_transform(modality)
        identity = torch.eye(16)
        assert torch.allclose(
            transform.T @ transform,
            identity,
            atol=2e-5,
            rtol=0.0,
        )
        assert torch.equal(transform, replay.modality_transform(modality))


def test_aligned_renderer_requires_one_shared_width() -> None:
    with pytest.raises(ValueError, match="shared width"):
        SyntheticManifoldRenderer(
            GraphSchema(summary_dim=8, context_dim=16, edge_dim=16),
            query_dim=16,
            geometry="orthogonal_aligned",
        )


@pytest.mark.parametrize(
    ("geometry", "should_pass"),
    [
        ("independent", False),
        ("shared_additive", True),
        ("orthogonal_aligned", True),
    ],
)
def test_unseen_symbol_identifiability_gate(
    geometry: str,
    should_pass: bool,
) -> None:
    report = run_renderer_identifiability(
        _renderer(geometry),
        config=IdentifiabilityProbeConfig(
            train_symbol_count=768,
            test_symbol_count=320,
            steps=240,
            batch_size=128,
            seed=73,
        ),
    )

    assert report.row_permutation_mismatches == 0
    assert report.passed is should_pass
    if should_pass:
        assert report.minimum_auroc >= 0.99
        assert report.minimum_top1_at_64 >= 0.95
        assert report.minimum_top1_at_256 >= 0.85
    else:
        assert report.macro_auroc < 0.60
        assert report.macro_top1_at_64 < 0.10
