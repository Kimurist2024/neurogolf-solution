#!/usr/bin/env python3
"""Run every numeric lower history lead in an isolated subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main() -> None:
    inventory = json.loads((HERE / "history_inventory.json").read_text())
    process_rows: list[dict[str, object]] = []
    for index, row in enumerate(inventory["retained"], 1):
        task = int(row["task"])
        candidate = ROOT / row["candidate"]
        label = f"task{task:03d}_history_{index:02d}_{row['sha256'][:10]}"
        destination = HERE / "evidence" / f"{label}.json"
        command = [
            sys.executable,
            str(HERE / "audit_one.py"),
            "--label",
            label,
            "--task",
            str(task),
            "--model",
            str(candidate),
            "--out",
            str(destination),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            result = {
                "task": task,
                "label": label,
                "candidate": row["candidate"],
                "sha256": row["sha256"],
                "static_cost": row["static_cost"],
                "exit_code": completed.returncode,
                "audit_json": str(destination.relative_to(ROOT)) if destination.is_file() else None,
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        except subprocess.TimeoutExpired as exc:
            result = {
                "task": task,
                "label": label,
                "candidate": row["candidate"],
                "sha256": row["sha256"],
                "static_cost": row["static_cost"],
                "exit_code": None,
                "timeout_seconds": 90,
                "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else None,
                "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else None,
            }
        process_rows.append(result)
        (HERE / "candidate_process_results.json").write_text(
            json.dumps({"completed": len(process_rows), "rows": process_rows}, indent=2) + "\n"
        )
        print(
            f"{index:02d}/{len(inventory['retained'])} task{task:03d} "
            f"static={row['static_cost']['cost']} exit={result.get('exit_code')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
