#!/usr/bin/env python3
"""Run independent random500 differential audits on historical task366 leads."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VALIDATOR = ROOT / "others/3/7907/neurogolf_team_validator_v1/ngolf_validator.py"
BASELINE = HERE / "baseline_task366.onnx"
CANDIDATES = {
    "historical_7985": ROOT / "others/2/1203/task366_improved.onnx",
    "historical_7916": ROOT / "others/2/1203/task366_further_improved.onnx",
}


def audit(item: tuple[str, Path]) -> dict[str, object]:
    label, candidate = item
    output = HERE / f"external_{label}_random500.json"
    log = HERE / f"external_{label}_random500.log"
    command = [
        sys.executable,
        str(VALIDATOR),
        "validate-task",
        "--task",
        "366",
        "--candidate-model",
        str(candidate),
        "--baseline-model",
        str(BASELINE),
        "--data-dir",
        str(ROOT / "inputs/neurogolf-2026"),
        "--data-zip",
        str(ROOT / "inputs/neurogolf-2026.zip"),
        "--random-cases",
        "500",
        "--seed",
        "799913366",
        "--allow-random-mismatch",
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
    result: dict[str, object] = {
        "label": label,
        "candidate": str(candidate.relative_to(ROOT)),
        "returncode": completed.returncode,
        "output": str(output.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        result["decision"] = payload.get("decision")
        result["differential"] = payload.get("differential")
    return result


def main() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(audit, CANDIDATES.items()))
    (HERE / "historical_random500_summary.json").write_text(
        json.dumps({"random_cases": 500, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
