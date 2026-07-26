"""Report bounded status for the detached Spider v0.1 Colab process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


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


def command_output(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    return output or None

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
active_run = job_status.get("active_run")
active_run_path = (
    run_directories / active_run
    if isinstance(active_run, str)
    else None
)
active_files = (
    {
        str(path.relative_to(active_run_path)): path.stat().st_size
        for path in sorted(active_run_path.rglob("*"))
        if path.is_file()
    }
    if active_run_path is not None and active_run_path.is_dir()
    else {}
)
history_tail: list[dict[str, object]] = []
history_path = (
    active_run_path / "history.jsonl"
    if active_run_path is not None
    else None
)
if history_path is not None and history_path.is_file():
    lines = history_path.read_text().splitlines()[-3:]
    history_tail = [json.loads(line) for line in lines if line.strip()]

gpu_status = command_output(
    [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
)
process_status = command_output(
    ["ps", "-o", "etimes=,pcpu=,pmem=,rss=,cmd=", "-p", str(pid)]
)
process_table = command_output(
    ["ps", "-eo", "pid=,ppid=,etimes=,pcpu=,pmem=,rss=,cmd="]
)
descendant_rows: list[str] = []
if process_table is not None:
    rows = [row.strip() for row in process_table.splitlines() if row.strip()]
    parsed_rows: list[tuple[int, int, str]] = []
    for row in rows:
        fields = row.split(maxsplit=2)
        if len(fields) < 3:
            continue
        try:
            row_pid, parent_pid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        parsed_rows.append((row_pid, parent_pid, row))
    descendant_ids = {pid}
    changed = True
    while changed:
        changed = False
        for row_pid, parent_pid, _ in parsed_rows:
            if parent_pid in descendant_ids and row_pid not in descendant_ids:
                descendant_ids.add(row_pid)
                changed = True
    descendant_rows = [
        row
        for row_pid, _, row in parsed_rows
        if row_pid in descendant_ids
    ]
print(
    json.dumps(
        {
            "active_files": active_files,
            "completed_metric_files": completed_metric_files,
            "gpu_status": gpu_status,
            "history_tail": history_tail,
            "job_status": job_status,
            "log_tail": log_tail,
            "pid": pid,
            "process_alive": process_alive,
            "process_status": process_status,
            "process_tree": descendant_rows,
        },
        sort_keys=True,
    )
)
