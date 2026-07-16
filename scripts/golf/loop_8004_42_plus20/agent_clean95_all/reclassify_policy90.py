#!/usr/bin/env python3
"""Confirm historical 90--95% rows while preserving clean-policy exclusions."""

from __future__ import annotations

import json
from pathlib import Path

from screen_all import HERE, ROOT, fresh_dual


SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen.json"


def main() -> int:
    old = json.loads(SOURCE.read_text())
    selected: dict[int, list[dict[str, object]]] = {}
    for row in old["rows"]:
        fresh = row.get("fresh_dual")
        if not isinstance(fresh, dict):
            continue
        task = int(row["task"])
        # task009 b265 is already adopted and every other task009 row is no
        # cheaper than that new current member.
        if task == 9:
            continue
        rates = [mode["right"] / 500 for mode in fresh.values()]
        if min(rates) < 0.90 or min(rates) >= 0.95:
            continue
        path = ROOT / row["isolated_candidate"]
        selected.setdefault(task, []).append({**row, "data": path.read_bytes()})

    output = {
        "source": str(SOURCE.relative_to(ROOT)),
        "selection": "old fresh500 dual rate in [0.90, 0.95), task009 excluded",
        "clean_policy": "private-zero catalog remains a hard rejection",
        "count": sum(len(rows) for rows in selected.values()),
        "tasks": {},
    }
    for task, rows in sorted(selected.items()):
        task_out = {
            "catalog_excluded": True,
            "reason": "private_zero_catalog_task",
            "initial_fresh500": {
                row["sha256"]: row["fresh_dual"] for row in rows
            },
            "seeds": {},
        }
        for seed_index, seed_base in enumerate((84_000_000, 85_000_000), 1):
            report = fresh_dual(task, rows, 5000, seed_base + task)
            task_out["seeds"][f"seed{seed_index}"] = report
            (HERE / "evidence" / f"task{task:03d}_policy90_reclass_5000_seed{seed_index}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            (HERE / "policy90_reclassification.json").write_text(
                json.dumps(output, indent=2) + "\n"
            )
        output["tasks"][str(task)] = task_out

    # Add task records after their runs so an interrupted task is never shown
    # as complete. The final write is authoritative.
    output["tasks"] = {}
    for task, rows in sorted(selected.items()):
        seeds = {}
        for seed_index in (1, 2):
            seeds[f"seed{seed_index}"] = json.loads(
                (HERE / "evidence" / f"task{task:03d}_policy90_reclass_5000_seed{seed_index}.json").read_text()
            )
        output["tasks"][str(task)] = {
            "catalog_excluded": True,
            "reason": "private_zero_catalog_task",
            "initial_fresh500": {row["sha256"]: row["fresh_dual"] for row in rows},
            "seeds": seeds,
        }
    (HERE / "policy90_reclassification.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
