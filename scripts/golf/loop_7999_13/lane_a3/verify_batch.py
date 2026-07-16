#!/usr/bin/env python3
"""Run strict repository verification for a directory of task ONNX files."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (13, 88, 157, 182, 191, 280, 330)


def verify(task: int, model: Path, count: int) -> dict[str, object]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "verify_fix.py"),
        "--task",
        str(task),
        "--onnx",
        str(model),
        "--k",
        str(count),
        "--min-fresh-rate",
        "1.0",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    combined = result.stdout
    match = re.search(r"\[\s*\{.*?\}\s*\]", result.stdout, re.DOTALL)
    parsed = json.loads(match.group(0))[0] if match else {}
    return {
        "task": task,
        "path": str(model.relative_to(ROOT)),
        "exit_code": result.returncode,
        "result": parsed,
        "runtime_exception": "RUNTIME_EXCEPTION" in combined,
        "stdout": result.stdout,
        "stderr": "[suppressed: ORT shape warnings are represented by verifier decision/runtime fields]",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--k", type=int, default=5000)
    parser.add_argument("--tasks", default=",".join(map(str, TASKS)))
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    tasks = tuple(int(item) for item in arguments.tasks.split(",") if item)
    models = arguments.models if arguments.models.is_absolute() else ROOT / arguments.models
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(verify, task, models / f"task{task:03d}.onnx", arguments.k): task
            for task in tasks
        }
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    rows.sort(key=lambda row: int(row["task"]))
    summary = [{key: value for key, value in row.items() if key not in ("stdout", "stderr")} for row in rows]
    (HERE / f"verify_{arguments.label}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    log = "\n".join(
        f"=== task{row['task']:03d} exit={row['exit_code']} ===\n{row['stdout']}\n{row['stderr']}"
        for row in rows
    )
    (HERE / f"verify_{arguments.label}.log").write_text(log, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
