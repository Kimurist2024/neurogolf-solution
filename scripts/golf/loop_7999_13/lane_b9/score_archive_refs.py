#!/usr/bin/env python3
"""Profile retained archive candidates for the B9 target set.

This is deliberately read-only with respect to submissions.  Each candidate is
scored in a fresh subprocess because some historical models can crash or hang
inside ONNX Runtime.
"""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = ROOT / "scripts/golf/loop_7999_13/lane_archive_zip_sweep"
TARGETS = {156, 182, 216, 237, 238, 284, 379}


def score(task: int, path: Path) -> dict[str, object]:
    command = [
        sys.executable,
        str(ROOT / "scripts/golf/verify_candidate_timeout.py"),
        "--task",
        str(task),
        "--onnx",
        str(path),
        "--timeout",
        "45",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    parsed: dict[str, object] = {}
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "exit_code": completed.returncode,
        "result": parsed,
        "stderr_tail": completed.stderr[-2000:],
    }


def main() -> None:
    inventory = json.loads((ARCHIVE / "inventory.json").read_text())
    jobs: list[tuple[int, Path]] = []
    for key, rows in inventory["retained"].items():
        task = int(key)
        if task not in TARGETS:
            continue
        for row in rows:
            jobs.append((task, ROOT / row["path"]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda job: score(*job), jobs))
    rows.sort(key=lambda row: (int(row["task"]), str(row["path"])))
    (HERE / "archive_actual_scores.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    concise = [
        {
            "task": row["task"],
            "path": row["path"],
            "exit_code": row["exit_code"],
            "result": row["result"],
        }
        for row in rows
    ]
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
