#!/usr/bin/env python3
"""Export the selected training checkpoint as portable inference weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hippocampus.sre import PackedSRECanonicalRetriever, SREModelConfig  # noqa: E402


MANIFEST = ROOT / "artifacts/spider_v0_8/SELECTED_CHECKPOINT.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    checkpoint_path = ROOT / manifest["checkpoint_path"]
    if _sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("selected training checkpoint hash disagrees")
    config = json.loads((ROOT / manifest["config"]).read_text())
    model = PackedSRECanonicalRetriever(SREModelConfig(**config["model"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    output_path = ROOT / manifest["portable_weights_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: value.detach().contiguous()
            for name, value in model.state_dict().items()
        },
        output_path,
        metadata={
            "format": "spider-v0.8-sre-inference-weights-v1",
            "training_checkpoint_sha256": manifest["checkpoint_sha256"],
        },
    )
    print(json.dumps({"path": str(output_path), "sha256": _sha256(output_path)}))


if __name__ == "__main__":
    main()
