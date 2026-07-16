#!/usr/bin/env python3
"""Condense existing repository-wide scans for the seven C9 tasks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (310, 330, 340, 354, 361, 364, 368)
SOURCES = (
    ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json",
    ROOT / "scripts/golf/loop_7999_13/lane_headroom/scan_results.json",
    ROOT / "scripts/golf/scratch_codex_plus10/wave3_c/scan_results.json",
)


def task_number(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("task") and value[4:].isdigit():
        return int(value[4:])
    return None


def compact(row: dict[str, object], source: Path) -> dict[str, object]:
    sources = row.get("sources") or [""]
    return {
        "scan": str(source.relative_to(ROOT)),
        "sha256": row.get("sha256"),
        "source": sources[0],
        "stage": row.get("stage") or row.get("status"),
        "actual_cost": row.get("actual_screen_cost") if row.get("actual_screen_cost") is not None else row.get("cost"),
        "static_cost_floor": row.get("static_cost_floor"),
        "correct": row.get("correct"),
        "reason": row.get("reason"),
    }


def main() -> None:
    collected: dict[int, list[tuple[dict[str, object], Path]]] = {task: [] for task in TASKS}
    for source in SOURCES:
        payload = json.loads(source.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            task = task_number(row.get("task"))
            if task in collected:
                collected[task].append((row, source))
    result: dict[str, object] = {}
    for task, rows in collected.items():
        stages = Counter((row.get("stage") or row.get("status")) for row, _ in rows)
        actual = [
            compact(row, source)
            for row, source in rows
            if row.get("actual_screen_cost") is not None or row.get("cost") is not None
        ]
        actual.sort(key=lambda item: int(item["actual_cost"]))
        static = [
            compact(row, source)
            for row, source in rows
            if row.get("static_cost_floor") is not None
        ]
        static.sort(key=lambda item: int(item["static_cost_floor"]))
        result[str(task)] = {
            "rows": len(rows),
            "stage_counts": dict(stages),
            "lowest_actual": actual[:10],
            "lowest_static_floor": static[:10],
        }
    (HERE / "history_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
