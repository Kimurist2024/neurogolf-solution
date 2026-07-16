#!/usr/bin/env python3
"""Run the full strict gate for one pre-screened 8019.75 candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent / "focus"
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
    "restart8019_75_focus_main",
    ROOT / "scripts/golf/restart8019_75_lane_high/worker.py",
)
BASE = MAIN.BASE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    args = parser.parse_args()
    task = args.task
    source = args.onnx if args.onnx.is_absolute() else ROOT / args.onnx
    HERE.mkdir(parents=True, exist_ok=True)

    ledger_cost = MAIN.COSTS[task]
    BASE.HERE = HERE
    BASE.BAND = ((task, ledger_cost),)
    BASE.ELIGIBLE = (task,)
    BASE.COSTS = MAIN.COSTS
    BASE.PRIVATE_ZERO_OR_UNSOUND = set()
    BASE.EXPLICIT_LATEST_LB_BLACK = set()
    BASE.CHANGED_FROM_8011_05 = {task}

    worker = BASE.Worker(0)
    MAIN.reprofile_authority(worker)
    worker.consider(task, source.read_bytes(), {
        "name": source.name,
        "family": "focused_new_drop",
        "detail": "focused continuation after slow high-cost audit",
        "source": str(source),
    })
    pre = worker.full_audit()
    gates = []
    accepted = []
    for row in pre:
        path = ROOT / row["saved_path"]
        gate = MAIN.strict_gate(path, task, int(row["authority_cost"]))
        gates.append(gate)
        if gate["pass"]:
            row["strict_gate"] = gate
            accepted.append(row)
    payload = {
        "task": task,
        "source": str(source),
        "authority_cost": worker.task_rows[task]["authority_cost"],
        "pre_strict": pre,
        "strict_gates": gates,
        "finalists": accepted,
        "task_row": worker.task_rows[task],
    }
    output = HERE / f"task{task:03d}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "task": task,
        "strict_winners": [
            {"cost": row["candidate_cost"], "gain": row["score_gain"],
             "path": row["saved_path"]}
            for row in accepted
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
