#!/usr/bin/env python3
"""Run an independent dual-ORT fresh-5000 confirmation for stage-500 survivors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescreen_all import HERE, fresh_dual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    report = json.loads((HERE / "rescreen.json").read_text())
    survivors = report["fresh500_survivors"]
    tasks: dict[int, list[dict[str, object]]] = {}
    for row in survivors:
        path = Path(row["isolated_candidate"])
        if not path.is_absolute():
            path = HERE.parents[3] / path
        tasks.setdefault(int(row["task"]), []).append(
            {**row, "data": path.read_bytes()}
        )
    output = {
        "independent_from_fresh500": True,
        "count_per_mode": 5000,
        "seed_offset": args.seed_offset,
        "tasks": {},
    }
    for task, rows in sorted(tasks.items()):
        result = fresh_dual(task, rows, 5000, 80_046_000 + task + args.seed_offset)
        output["tasks"][str(task)] = result
        suffix = "" if args.seed_offset == 0 else f"_seed{args.seed_offset}"
        (HERE / "evidence" / f"task{task:03d}_fresh_dual_5000{suffix}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        (HERE / f"fresh5000_confirmation{suffix}.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
