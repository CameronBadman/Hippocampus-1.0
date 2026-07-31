#!/usr/bin/env python3
"""Render the guarded Better Colab launch cell for Spider v0.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/colab_spider_v0_3_worker.py"
REPOSITORY_RAW = (
    "https://raw.githubusercontent.com/CameronBadman/"
    "Hippocampus-1.0"
)
DATASET_SHA256 = (
    "0ed8e27ec44f3773f76b79f1947526f33ba233556b7db91fef04dcb647e5409d"
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def render_launch_source() -> str:
    source_commit = _git("rev-parse", "HEAD")
    worker_relative = str(WORKER.relative_to(ROOT))
    if _git("status", "--porcelain", "--", worker_relative):
        raise RuntimeError("Colab worker has uncommitted changes")
    tracked = _git("ls-files", "--error-unmatch", worker_relative)
    if tracked != worker_relative:
        raise RuntimeError("Colab worker is not tracked")
    worker_sha256 = hashlib.sha256(WORKER.read_bytes()).hexdigest()
    worker_url = f"{REPOSITORY_RAW}/{source_commit}/{worker_relative}"
    run_spec = {
        "source_commit": source_commit,
        "dataset_sha256": DATASET_SHA256,
        "phase": "all",
    }
    return "\n".join(
        (
            "import hashlib",
            "import urllib.request",
            "",
            f"RUN_SPEC = {json.dumps(run_spec, sort_keys=True)}",
            f"WORKER_URL = {worker_url!r}",
            f"EXPECTED_SHA256 = {worker_sha256!r}",
            "source = urllib.request.urlopen(WORKER_URL, timeout=60).read()",
            "if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:",
            "    raise RuntimeError('Spider v0.3 worker hash mismatch')",
            "exec(compile(source, WORKER_URL, 'exec'), globals())",
            "",
        )
    )


if __name__ == "__main__":
    print(render_launch_source(), end="")
