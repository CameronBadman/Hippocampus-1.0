from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    del config
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA is not visible to this test process")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)

