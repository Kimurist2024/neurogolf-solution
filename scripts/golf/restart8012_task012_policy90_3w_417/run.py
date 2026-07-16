#!/usr/bin/env python3
"""Run exactly three task012 search subprocesses and merge their evidence."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WORKER = HERE / "worker.py"
PARENT = ROOT / (
    "scripts/golf/root_task012_h8w8_policy90_272/candidates/"
    "task012_h8w8_policy90.onnx"
)


def run_worker(index: int) -> dict:
    completed = subprocess.run(
        [sys.executable, str(WORKER), str(index)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "worker": index,
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def main() -> None:
    # The task brief requires exactly three internal workers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        runs = list(executor.map(run_worker, range(3)))
    (HERE / "worker_runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    failed = [row for row in runs if row["returncode"] != 0]
    if failed:
        print(json.dumps(runs, indent=2))
        raise SystemExit(1)
    evidence = [
        json.loads((HERE / "worker0_dense.json").read_text()),
        json.loads((HERE / "worker1_nobias.json").read_text()),
        json.loads((HERE / "worker2_alternatives.json").read_text()),
    ]
    result = {
        "task": 12,
        "lane": "restart8012_task012_policy90_3w_417",
        "internal_workers": 3,
        "authority_candidate": {
            "path": str(PARENT.relative_to(ROOT)),
            "sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            "cost": 650,
        },
        "threshold": 0.90,
        "cost_target": "strictly below 650",
        "workers": evidence,
        "finalists": [],
        "decision": "NO_SUB650_POLICY90_FINALIST",
        "root_submission_modified": False,
        "root_scores_modified": False,
        "others_modified": False,
    }
    if any(row["policy90_found"] for row in evidence):
        result["decision"] = "REVIEW_REQUIRED_WORKER_REPORTED_POLICY90"
    (HERE / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": result["decision"], "workers": 3}, indent=2))


if __name__ == "__main__":
    main()
