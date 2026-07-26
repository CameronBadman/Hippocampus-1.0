"""Report bounded status for the detached Spider v0.1 Colab process."""

from __future__ import annotations

import json
import os
from pathlib import Path


OUTPUT = Path("/content/spider-v01-colab-5k")
launch_path = OUTPUT / "LAUNCH.json"
if not launch_path.is_file():
    raise FileNotFoundError("training launch record is missing")
launch = json.loads(launch_path.read_text())
pid = int(launch["pid"])
try:
    os.kill(pid, 0)
except ProcessLookupError:
    process_alive = False
else:
    process_alive = True

job_status_path = OUTPUT / "JOB_STATUS.json"
job_status = (
    json.loads(job_status_path.read_text())
    if job_status_path.is_file()
    else {"state": "starting"}
)
log_path = OUTPUT / "training.log"
if log_path.is_file():
    with log_path.open("rb") as handle:
        handle.seek(max(0, log_path.stat().st_size - 8_192))
        log_tail = handle.read().decode("utf-8", errors="replace")
else:
    log_tail = ""

run_directories = OUTPUT / "runs"
completed_metric_files = (
    sorted(
        str(path.relative_to(OUTPUT))
        for path in run_directories.glob("*/metrics.json")
    )
    if run_directories.is_dir()
    else []
)
print(
    json.dumps(
        {
            "completed_metric_files": completed_metric_files,
            "job_status": job_status,
            "log_tail": log_tail,
            "pid": pid,
            "process_alive": process_alive,
        },
        sort_keys=True,
    )
)
