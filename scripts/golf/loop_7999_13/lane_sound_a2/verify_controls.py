#!/usr/bin/env python3
"""Run the repository strict verifier for every retained sound control."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (9, 77, 173)


def main() -> None:
    sections: list[str] = []
    failures: list[int] = []
    for task in TASKS:
        model = HERE / "sound_controls" / f"task{task:03d}.onnx"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "verify_fix.py"),
            "--task",
            str(task),
            "--onnx",
            str(model),
            "--k",
            "5000",
            "--min-fresh-rate",
            "1.0",
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        sections.append(
            f"=== task{task:03d} exit={result.returncode} ===\n"
            f"{result.stdout}\n{result.stderr}"
        )
        if result.returncode != 0 or "ADOPT=1/1" not in result.stdout + result.stderr:
            failures.append(task)
    log = "\n".join(sections)
    (HERE / "verify_sound_controls_fresh5000.log").write_text(log, encoding="utf-8")
    print(log)
    if failures:
        raise RuntimeError(f"strict verifier failed for tasks {failures}")


if __name__ == "__main__":
    main()
