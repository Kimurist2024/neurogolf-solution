#!/usr/bin/env python3
"""Run the broad mechanical exact-rewrite census on 8023.08 high-cost tasks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent / "deep_exact"
ROOT = HERE.parents[3]


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = import_path(
    "restart8023_08_deep_exact_base",
    ROOT / "scripts/golf/restart8012_exact_rewrite_3w_415/scan.py",
)

SCAN.HERE = HERE
SCAN.AUTHORITY = ROOT / "submission_base_8023.08.zip"
SCAN.AUTHORITY_SHA256 = "0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a"
SCAN.MIN_COST = 300
SCAN.MAX_COST = 100_000
SCAN.EXCLUDED = set(SCAN.EXCLUDED) | {132, 168, 226, 275, 345}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if SCAN.sha256(SCAN.AUTHORITY.read_bytes()) != SCAN.AUTHORITY_SHA256:
        raise RuntimeError("8023.08 authority SHA mismatch")
    official_costs = SCAN.costs()
    tasks = SCAN.selected_tasks(official_costs)
    partitions = [tasks[index::3] for index in range(3)]
    result = SCAN.worker(args.worker, partitions[args.worker], official_costs)
    finalists = [
        row for row in result["rows"] if row["status"] == "finalist_known_exact"
    ]
    print(json.dumps({
        "worker": args.worker,
        "summary": result["summary"],
        "finalists": [
            {
                "task": row["task"],
                "label": row["label"],
                "cost": row["official_profile"]["cost"],
                "gain": row["gain"],
                "path": row["path"],
                "sha256": row["candidate_sha256"],
            }
            for row in finalists
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
