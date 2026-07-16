#!/usr/bin/env python3
"""Fast 8019.75 screen: enumerate strict-lower known-exact survivors only."""

from __future__ import annotations

import argparse
import json

import worker as lane


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()

    candidate_worker = lane.BASE.Worker(args.worker)
    lane.reprofile_authority(candidate_worker)
    lane.scan_new_drops(candidate_worker)
    rows = []
    for task in candidate_worker.tasks:
        rows.extend(candidate_worker.task_rows[task]["screen_survivors"])
    payload = {
        "worker": args.worker,
        "tasks": len(candidate_worker.tasks),
        "survivors": rows,
        "counters": dict(candidate_worker.counters),
    }
    output = lane.HERE / f"screen_{args.worker}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "worker": args.worker,
        "survivors": [
            {
                "task": row["task"],
                "cost": row["candidate_cost"],
                "source": row.get("source"),
                "sha256": row["sha256"],
            }
            for row in rows
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
