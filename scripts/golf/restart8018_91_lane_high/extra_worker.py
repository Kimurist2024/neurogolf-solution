#!/usr/bin/env python3
"""Companion worker covering the remaining 8018.91 cost-400..500 tasks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent / "extra"
ROOT = HERE.parents[3]


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MAIN = import_path(
    "restart8018_91_high_main",
    ROOT / "scripts/golf/restart8018_91_lane_high/worker.py",
)
BASE = MAIN.BASE
BAND = (
    (8, 421),
    (184, 421),
    (333, 421),
    (268, 420),
    (112, 418),
    (134, 417),
    (377, 409),
    (354, 403),
)
ELIGIBLE = tuple(task for task, _cost in BAND)
COSTS = dict(BAND)

BASE.HERE = HERE
BASE.BAND = BAND
BASE.PRIVATE_ZERO_OR_UNSOUND = set()
BASE.EXPLICIT_LATEST_LB_BLACK = set()
BASE.ELIGIBLE = ELIGIBLE
BASE.COSTS = COSTS
BASE.CHANGED_FROM_8011_05 = set(ELIGIBLE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    payload = BASE.Worker(args.worker).run()
    accepted = []
    strict_gates = []
    for row in payload["finalists"]:
        path = ROOT / row["saved_path"]
        gate = MAIN.strict_gate(path, int(row["task"]), int(row["authority_cost"]))
        strict_gates.append(gate)
        if gate["pass"]:
            row["strict_gate"] = gate
            accepted.append(row)
    payload["pre_strict_gate_finalists"] = payload["finalists"]
    payload["strict_gates"] = strict_gates
    payload["finalists"] = accepted
    payload["absolute_admission_gate"] = (
        "local+official gold exact, strict structure, margin, fresh2000x2 100%"
    )
    output = HERE / f"worker_{args.worker}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "worker": args.worker,
        "tasks": payload["assigned_tasks"],
        "strict_winners": [
            {"task": r["task"], "cost": r["candidate_cost"],
             "gain": r["score_gain"], "path": r["saved_path"]}
            for r in accepted
        ],
        "elapsed": payload["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
