"""Launch one frozen Spider v0.1 replication as a detached Colab process."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request


SPEC = Path("/content/SPIDER_RUN_SPEC.json")
ACTIVE = Path("/content/SPIDER_ACTIVE_RUN.json")
REMOTE_WORKER = Path("/content/colab_spider_v0_1_single_run.py")
WORKER_URL = (
    "https://raw.githubusercontent.com/CameronBadman/Hippocampus-1.0/"
    "81ec4c3cfc340ffefca886823a7273452baa8f1d/"
    "scripts/colab_spider_v0_1_single_run.py"
)
EXPECTED_WORKER_SHA256 = (
    "6fe6b79507a50918f9134b33f8ca407f3f6b7f760e46626e0d410c982c9acf85"
)
EXPERIMENT_ID = re.compile(
    r"L-E00[45]-(?:recurrent|pooled)-s(?:1701|1802|1903)-5k"
)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


if not SPEC.is_file():
    raise FileNotFoundError("run specification is missing")
spec = json.loads(SPEC.read_text())
experiment_id = str(spec["experiment_id"])
if EXPERIMENT_ID.fullmatch(experiment_id) is None:
    raise ValueError(f"invalid frozen experiment ID: {experiment_id!r}")

if ACTIVE.exists():
    prior = json.loads(ACTIVE.read_text())
    prior_pid = int(prior["pid"])
    if process_exists(prior_pid):
        raise RuntimeError(f"training process {prior_pid} is already active")
    raise RuntimeError("a prior active-run record exists; refusing to replay")

output = Path("/content/spider-v01-colab-5k-isolated") / experiment_id
archive = Path(f"/content/{experiment_id}-result.tar.gz")
if output.exists() or archive.exists():
    raise FileExistsError("prior run output exists; refusing to replay")

source = urllib.request.urlopen(WORKER_URL, timeout=60).read()
digest = hashlib.sha256(source).hexdigest()
if digest != EXPECTED_WORKER_SHA256:
    raise RuntimeError(f"single-run worker hash mismatch: {digest}")
REMOTE_WORKER.write_bytes(source)

environment = dict(os.environ)
environment["PYTHONUNBUFFERED"] = "1"
log_path = Path(f"/content/{experiment_id}.log")
with log_path.open("wb") as log:
    process = subprocess.Popen(
        [sys.executable, str(REMOTE_WORKER), str(SPEC)],
        cwd="/content",
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

launch = {
    "archive": str(archive),
    "experiment_id": experiment_id,
    "launched_at": datetime.now(timezone.utc).isoformat(),
    "log_path": str(log_path),
    "pid": process.pid,
    "worker_sha256": digest,
    "worker_url": WORKER_URL,
}
ACTIVE.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
print(json.dumps(launch, sort_keys=True))
