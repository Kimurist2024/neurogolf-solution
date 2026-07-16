#!/usr/bin/env python3
"""Run the independent validator's random500 differential on the old 968 lead."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VALIDATOR = ROOT / "others/3/7907/neurogolf_team_validator_v1/ngolf_validator.py"
CANDIDATE = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task196_r07_static296.onnx"
OUTPUT = HERE / "external_historical_968_random500.json"
LOG = HERE / "external_historical_968_random500.log"


def main() -> None:
    command = [
        sys.executable,
        str(VALIDATOR),
        "validate-task",
        "--task",
        "196",
        "--candidate-model",
        str(CANDIDATE),
        "--baseline-model",
        str(HERE / "baseline_task196.onnx"),
        "--data-dir",
        str(ROOT / "inputs/neurogolf-2026"),
        "--data-zip",
        str(ROOT / "inputs/neurogolf-2026.zip"),
        "--random-cases",
        "500",
        "--seed",
        "196800263",
        "--allow-random-mismatch",
        "--out-json",
        str(OUTPUT),
    ]
    with LOG.open("wb") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=360,
            check=False,
        )
    payload = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    summary = {
        "returncode": completed.returncode,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "log": str(LOG.relative_to(ROOT)),
        "decision": payload.get("decision"),
        "differential": payload.get("differential"),
    }
    (HERE / "external_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
