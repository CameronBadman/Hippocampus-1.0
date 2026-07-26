"""Launch the frozen Spider v0.1 long run as one detached Colab process."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


OUTPUT = Path("/content/spider-v01-colab-5k")
REMOTE_SCRIPT = Path("/content/colab_spider_v0_1_long_run.py")
SOURCE_URL = (
    "https://raw.githubusercontent.com/CameronBadman/Hippocampus-1.0/"
    "33c5b37c136723cf91467c7b06485e7cf7a7f196/"
    "scripts/colab_spider_v0_1_long_run.py"
)
EXPECTED_SHA256 = (
    "ef066e806268de4439004f952551ca6c4311149396a445739ed3bc80d9d87434"
)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


OUTPUT.mkdir(parents=True, exist_ok=True)
launch_path = OUTPUT / "LAUNCH.json"
if launch_path.exists():
    prior = json.loads(launch_path.read_text())
    prior_pid = int(prior["pid"])
    if process_exists(prior_pid):
        raise RuntimeError(f"training process {prior_pid} is already active")
    raise RuntimeError("a prior launch record exists; refusing to replay")

source = urllib.request.urlopen(SOURCE_URL, timeout=60).read()
digest = hashlib.sha256(source).hexdigest()
if digest != EXPECTED_SHA256:
    raise RuntimeError(f"long-run source hash mismatch: {digest}")
REMOTE_SCRIPT.write_bytes(source)

environment = dict(os.environ)
environment["PYTHONUNBUFFERED"] = "1"
log_path = OUTPUT / "training.log"
with log_path.open("wb") as log:
    process = subprocess.Popen(
        [sys.executable, str(REMOTE_SCRIPT)],
        cwd="/content",
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

launch = {
    "launched_at": datetime.now(timezone.utc).isoformat(),
    "log_path": str(log_path),
    "pid": process.pid,
    "source_sha256": digest,
    "source_url": SOURCE_URL,
}
launch_path.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
print(json.dumps(launch, sort_keys=True))
