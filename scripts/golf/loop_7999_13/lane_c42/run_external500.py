#!/usr/bin/env python3
"""Run the independent team-validator random-500 gate for task379."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VALIDATOR = ROOT / "others" / "3" / "7907" / "neurogolf_team_validator_v1" / "ngolf_validator.py"


def main() -> int:
    output = HERE / "task379_qv_middle_rank2_external500.json"
    log = HERE / "task379_qv_middle_rank2_external500.log"
    command = [
        sys.executable,
        str(VALIDATOR),
        "validate-task",
        "--task",
        "379",
        "--candidate-model",
        str(HERE / "candidates" / "task379_qv_middle_rank2.onnx"),
        "--baseline-model",
        str(HERE / "baseline" / "task379.onnx"),
        "--data-dir",
        str(ROOT / "inputs" / "neurogolf-2026"),
        "--data-zip",
        str(ROOT / "inputs" / "neurogolf-2026.zip"),
        "--random-cases",
        "500",
        "--seed",
        "800263379042",
        "--out-json",
        str(output),
    ]
    with log.open("wb") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=360,
            check=False,
        )
    payload = json.loads(output.read_text()) if output.exists() else None
    summary = {
        "command": command,
        "returncode": completed.returncode,
        "output": str(output.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
        "decision": payload.get("decision") if payload else None,
        "differential": payload.get("differential") if payload else None,
    }
    summary_path = HERE / "task379_qv_middle_rank2_external500_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
