#!/usr/bin/env python3
"""Known/fresh-screen every retained archive candidate variant."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import re
from pathlib import Path

from strict_verify_batch import verify


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK_RE = re.compile(r"task(\d{3})_")


def parse_tasks(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.split(","):
        if not item:
            continue
        if "-" in item:
            left, right = map(int, item.split("-", 1))
            result.update(range(left, right + 1))
        else:
            result.add(int(item))
    return result


def costs() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open(newline="") as handle:
        return {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--glob", default="task*_r*.onnx")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = args.models if args.models.is_absolute() else ROOT / args.models
    selected_tasks = parse_tasks(args.tasks)
    candidates: list[tuple[int, Path]] = []
    for path in sorted(models.glob(args.glob)):
        match = TASK_RE.match(path.name)
        if match and int(match.group(1)) in selected_tasks:
            candidates.append((int(match.group(1)), path))

    base_costs = costs()
    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(verify, task, path, args.k, args.timeout): (task, path)
            for task, path in candidates
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            task = int(row["task"])
            result = row.get("result", {})
            candidate_cost = result.get("cost") if isinstance(result, dict) else None
            row["baseline_cost"] = base_costs[task]
            row["strictly_cheaper"] = bool(
                isinstance(candidate_cost, int) and candidate_cost < base_costs[task]
            )
            row["projected_gain"] = (
                math.log(base_costs[task] / candidate_cost)
                if isinstance(candidate_cost, int) and 0 < candidate_cost < base_costs[task]
                else 0.0
            )
            rows.append(row)
            print(
                f"task{task:03d} {Path(row['path']).name}: "
                f"decision={result.get('decision') if isinstance(result, dict) else None} "
                f"cost={candidate_cost}/{base_costs[task]} cheaper={row['strictly_cheaper']}",
                flush=True,
            )
    rows.sort(key=lambda row: (int(row["task"]), str(row["path"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    winners = [
        row for row in rows
        if row.get("strictly_cheaper")
        and isinstance(row.get("result"), dict)
        and row["result"].get("decision") == "ADOPT"
    ]
    print(json.dumps({
        "candidates": len(rows),
        "quick_winners": len(winners),
        "projected_gain": sum(float(row["projected_gain"]) for row in winners),
        "winner_paths": [row["path"] for row in winners],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
