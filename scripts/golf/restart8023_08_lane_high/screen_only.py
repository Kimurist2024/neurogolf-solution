#!/usr/bin/env python3
"""Fast survivor census for the 8023.08 high-cost lane."""

from __future__ import annotations

import argparse
import json

import worker as lane


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    lane.HERE.mkdir(parents=True, exist_ok=True)
    worker = lane.make_worker(args.worker)
    lane.screen(worker)
    rows = []
    for task in worker.tasks:
        rows.extend(worker.task_rows[task]["screen_survivors"])
    payload = {
        "worker": args.worker,
        "tasks": len(worker.tasks),
        "survivors": rows,
        "counters": dict(worker.counters),
    }
    (lane.HERE / f"screen_{args.worker}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "worker": args.worker,
        "survivors": [
            {
                "task": row["task"],
                "cost": row["candidate_cost"],
                "source": row.get("source"),
                "family": row.get("family"),
                "sha256": row["sha256"],
            }
            for row in rows
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
