"""Report bounded status for one isolated Spider v0.1 Colab run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


ACTIVE = Path("/content/SPIDER_ACTIVE_RUN.json")
if not ACTIVE.is_file():
    raise FileNotFoundError("active-run record is missing")
launch = json.loads(ACTIVE.read_text())
experiment_id = str(launch["experiment_id"])
pid = int(launch["pid"])
output = Path("/content/spider-v01-colab-5k-isolated") / experiment_id


def process_exists(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    return True


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
    text = completed.stdout.strip()
    return text or None


status_path = output / "JOB_STATUS.json"
job_status = (
    json.loads(status_path.read_text())
    if status_path.is_file()
    else {"state": "starting"}
)
log_path = Path(launch["log_path"])
if log_path.is_file():
    with log_path.open("rb") as handle:
        handle.seek(max(0, log_path.stat().st_size - 8_192))
        log_tail = handle.read().decode("utf-8", errors="replace")
else:
    log_tail = ""

files = (
    {
        str(path.relative_to(output)): path.stat().st_size
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    if output.is_dir()
    else {}
)
history_tail: list[dict[str, object]] = []
history_path = output / "run" / "history.jsonl"
if history_path.is_file():
    history_tail = [
        json.loads(line)
        for line in history_path.read_text().splitlines()[-3:]
        if line.strip()
    ]

process_table = command_output(
    ["ps", "-eo", "pid=,ppid=,etimes=,pcpu=,pmem=,rss=,cmd="]
)
process_tree: list[str] = []
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
    process_tree = [
        row
        for row_pid, _, row in parsed_rows
        if row_pid in descendant_ids
    ]

archive = Path(launch["archive"])
print(
    json.dumps(
        {
            "archive_bytes": archive.stat().st_size if archive.is_file() else 0,
            "archive_ready": archive.is_file(),
            "experiment_id": experiment_id,
            "files": files,
            "gpu_status": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "history_tail": history_tail,
            "job_status": job_status,
            "log_tail": log_tail,
            "pid": pid,
            "process_alive": process_exists(pid),
            "process_tree": process_tree,
        },
        sort_keys=True,
    )
)
