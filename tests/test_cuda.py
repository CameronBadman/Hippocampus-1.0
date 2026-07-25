from __future__ import annotations

import pytest
import torch

from hippocampus import (
    GraphSchema,
    PackConfig,
    RaggedManifoldComponents,
    TopologyComponent,
    VarlenManifoldBatch,
    pack_graph_from_topology,
    pack_topology,
    segment_logsumexp,
    segment_softmax,
    segment_sum,
)


pytestmark = pytest.mark.cuda


def _cuda_topology(schema: GraphSchema):
    cpu = pack_topology(
        [TopologyComponent(2, [0], [1], True, schema)]
    )
    return cpu.pin_memory().to("cuda", non_blocking=True)


def test_cuda_default_preserves_device_and_fp16() -> None:
    schema = GraphSchema(2, 2, 2)
    topology = _cuda_topology(schema)
    graph = pack_graph_from_topology(
        topology,
        torch.ones(2, 2, device="cuda", dtype=torch.float16),
        RaggedManifoldComponents(
            torch.ones(1, 2, device="cuda", dtype=torch.float16),
            torch.tensor([0, 0, 1], device="cuda"),
        ),
        torch.ones(1, 2, device="cuda", dtype=torch.float16),
    )
    assert graph.device == torch.device("cuda:0")
    assert graph.dtype == torch.float16


def test_cuda_bf16_and_pinned_staging_are_differentiable() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support BF16")
    schema = GraphSchema(2, 2, 2)
    topology = _cuda_topology(schema)
    summaries = torch.randn(2, 2, requires_grad=True)
    contexts = torch.randn(1, 2, requires_grad=True)
    edges = torch.randn(1, 2, requires_grad=True)
    graph = pack_graph_from_topology(
        topology,
        summaries,
        RaggedManifoldComponents(contexts, [0, 0, 1]),
        edges,
        pack_config=PackConfig.cuda_bf16(pin_cpu_staging=True),
    )
    assert graph.dtype == torch.bfloat16
    assert graph.resolved_pack_config.staging_policy == "pinned_cpu"
    (
        graph.summaries.values.float().sum()
        + graph.contexts.values.float().sum()
        + graph.edges.values.float().sum()
    ).backward()
    assert summaries.grad is not None
    assert contexts.grad is not None
    assert edges.grad is not None


def test_cuda_auto_layout_selects_varlen_only_with_consumer_support() -> None:
    schema = GraphSchema(2, 2, 2)
    topology = _cuda_topology(schema)
    graph = pack_graph_from_topology(
        topology,
        torch.ones(2, 2, device="cuda"),
        RaggedManifoldComponents(
            torch.ones(3, 2, device="cuda"),
            torch.tensor([0, 1, 3], device="cuda"),
        ),
        torch.ones(1, 2, device="cuda"),
    )
    gathered = graph.contexts.gather(
        torch.tensor([0, 1], device="cuda")
    )
    supported = gathered.layout("auto", supports_cuda_varlen=True)
    unsupported = gathered.layout("auto", supports_cuda_varlen=False)
    assert isinstance(supported, VarlenManifoldBatch)
    assert not isinstance(unsupported, VarlenManifoldBatch)


def test_cuda_segmented_fast_and_deterministic_modes() -> None:
    values = torch.randn(
        1_024,
        8,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    offsets = torch.tensor(
        [0, 257, 257, 640, 1_024],
        dtype=torch.int32,
        device="cuda",
    )
    summed = segment_sum(values, offsets)
    lse = segment_logsumexp(values, offsets)
    softmax = segment_softmax(values, offsets)
    assert summed.values.dtype == torch.bfloat16
    assert summed.valid_mask.tolist() == [True, False, True, True]
    (summed.values.float().sum() + lse.values.float().sum()).backward()
    assert values.grad is not None
    assert torch.isfinite(softmax.values).all()

    torch.use_deterministic_algorithms(True)
    try:
        first = segment_logsumexp(
            values.detach(), offsets, execution_policy="deterministic"
        )
        second = segment_logsumexp(
            values.detach(), offsets, execution_policy="deterministic"
        )
        assert torch.equal(first.values, second.values)
    finally:
        torch.use_deterministic_algorithms(False)

