#!/usr/bin/env python3
"""Reclassify clean fresh-500 rows at 90% and confirm with two 5k seeds."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py"
SPEC = importlib.util.spec_from_file_location("mid20_84_confirm", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load exhaustive scanner")
SCANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCANNER)
SCANNER.HERE = HERE


def passed(stats: dict, count: int, threshold: float) -> bool:
    return all(
        int(item.get("right", 0)) / count >= threshold
        and int(item.get("right", 0)) + int(item.get("wrong", 0)) == count
        and int(item.get("errors", 0)) == 0
        and not item.get("session_error")
        for item in stats.values()
    )


def main() -> int:
    screen = json.loads((HERE / "rescreen.json").read_text())
    leads = []
    for row in screen["rows"]:
        stats = row.get("fresh_dual")
        path = row.get("isolated_candidate")
        if stats and path and passed(stats, 500, 0.90):
            leads.append({**row, "data": (ROOT / path).read_bytes()})

    confirmations = []
    for task in sorted({int(row["task"]) for row in leads}):
        task_rows = [row for row in leads if int(row["task"]) == task]
        for seed in (84_005_000 + task, 184_005_000 + task):
            confirmations.append(SCANNER.fresh_dual(task, task_rows, 5_000, seed))

    by_key = {}
    for report in confirmations:
        for digest, stats in report["candidates"].items():
            by_key.setdefault((int(report["task"]), digest), []).append(
                {"seed": report["seed"], "stats": stats, "pass90": passed(stats, 5_000, 0.90)}
            )
    survivors = []
    for row in leads:
        checks = by_key.get((int(row["task"]), row["sha256"]), [])
        if len(checks) == 2 and all(item["pass90"] for item in checks):
            survivors.append(
                {
                    "task": row["task"],
                    "sha256": row["sha256"],
                    "path": row["isolated_candidate"],
                    "current_cost": row["current_actual_cost"],
                    "candidate_cost": row["actual_cost"],
                    "gain": row["gain"],
                    "checks": checks,
                }
            )
    result = {
        "threshold": 0.90,
        "fresh500_leads": [
            {"task": row["task"], "sha256": row["sha256"], "path": row["isolated_candidate"]}
            for row in leads
        ],
        "confirmations": confirmations,
        "survivors": survivors,
    }
    (HERE / "policy90_confirmation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fresh500_leads": len(leads), "survivors": survivors}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
