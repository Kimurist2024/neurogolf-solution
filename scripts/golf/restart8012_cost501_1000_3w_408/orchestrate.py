#!/usr/bin/env python3
"""Launch the three disjoint cost-band workers concurrently."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> int:
    processes = []
    logs = []
    for worker in range(3):
        log = (HERE / f"worker_{worker}.log").open("w", encoding="utf-8")
        logs.append(log)
        processes.append(subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), str(HERE / "worker.py"), "--worker", str(worker)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
        ))
    codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    for worker, code in enumerate(codes):
        print(f"worker {worker}: exit {code}")
    return max(codes)


if __name__ == "__main__":
    raise SystemExit(main())
