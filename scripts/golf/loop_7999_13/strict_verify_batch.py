#!/usr/bin/env python3
"""Run repository strict known+fresh verification for a candidate task set."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def verify(task: int, model: Path, count: int, timeout: float) -> dict[str, object]:
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
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "task": task,
            "path": str(model.relative_to(ROOT)),
            "exit_code": None,
            "result": {},
            "runtime_exception": False,
            "timed_out": True,
            "stdout": exc.stdout or "",
            "stderr_tail": (exc.stderr or "")[-4000:],
        }
    match = re.search(r"\[\s*\{.*?\}\s*\]", result.stdout, re.DOTALL)
    parsed = json.loads(match.group(0))[0] if match else {}
    return {
        "task": task,
        "path": str(model.relative_to(ROOT)),
        "exit_code": result.returncode,
        "result": parsed,
        "runtime_exception": "RUNTIME_EXCEPTION" in (result.stdout + result.stderr),
        "stdout": result.stdout,
        "stderr_tail": result.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--k", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--pattern",
        default="task{task:03d}.onnx",
        help="filename pattern relative to --models; receives task=ID",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = args.models if args.models.is_absolute() else ROOT / args.models
    tasks: list[int] = []
    for item in args.tasks.split(","):
        if not item:
            continue
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            tasks.extend(range(start, end + 1))
        else:
            tasks.append(int(item))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        selected: dict[int, Path] = {}
        for task in tasks:
            rendered = args.pattern.format(task=task)
            matches = sorted(models.glob(rendered)) if any(char in rendered for char in "*?[") else []
            selected[task] = matches[0] if matches else models / rendered
        futures = {
            executor.submit(
                verify, task, selected[task], args.k, args.timeout
            ): task
            for task in tasks
        }
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    rows.sort(key=lambda row: int(row["task"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    concise = [
        {key: value for key, value in row.items() if key not in ("stdout", "stderr_tail")}
        for row in rows
    ]
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
