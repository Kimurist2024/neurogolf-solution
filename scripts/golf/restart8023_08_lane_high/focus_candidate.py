#!/usr/bin/env python3
"""Run the complete strict gate for one 8023.08 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import worker as lane


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    args = parser.parse_args()
    task = args.task
    source = args.onnx if args.onnx.is_absolute() else lane.ROOT / args.onnx
    focus = lane.HERE / "focus"
    focus.mkdir(parents=True, exist_ok=True)

    ledger_cost = lane.COSTS[task]
    lane.BASE.HERE = focus
    lane.BASE.BAND = ((task, ledger_cost),)
    lane.BASE.ELIGIBLE = (task,)
    lane.BASE.COSTS = lane.COSTS
    lane.BASE.CHANGED_FROM_8011_05 = {task}
    worker = lane.BASE.Worker(0)
    lane.OLD.reprofile_authority(worker)
    worker.consider(task, source.read_bytes(), {
        "name": source.name,
        "family": "focused_8023_new_drop",
        "detail": "focused strict audit against immutable 8023.08 authority",
        "source": str(source),
    })
    pre = worker.full_audit()
    gates = []
    accepted = []
    for row in pre:
        path = lane.ROOT / row["saved_path"]
        gate = lane.OLD.strict_gate(path, task, int(row["authority_cost"]))
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
    (focus / f"task{task:03d}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "task": task,
        "strict_winners": [
            {
                "cost": row["candidate_cost"],
                "gain": row["score_gain"],
                "path": row["saved_path"],
            }
            for row in accepted
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
