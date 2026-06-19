#!/usr/bin/env python3
"""Build the overnight golf factory queue from current task costs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLF_DOCS = REPO_ROOT / "docs" / "golf"
CURRENT_COSTS = GOLF_DOCS / "current_costs.json"
TASK_HASH_MAP = GOLF_DOCS / "task_hash_map.json"
FACTORY_DONE = GOLF_DOCS / "factory_done.json"
QUEUE_PATH = GOLF_DOCS / "queue.json"
MIN_COST_EXCLUSIVE = 2500


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _task_id(value: Any) -> int:
    return int(str(value).removeprefix("task"))


def _done_tasks(payload: Any) -> set[int]:
    if isinstance(payload, list):
        return {_task_id(item) for item in payload}
    if isinstance(payload, dict):
        if isinstance(payload.get("tasks"), list):
            return {_task_id(item) for item in payload["tasks"]}
        return {_task_id(key) for key, value in payload.items() if value}
    return set()


def _cost(value: Any) -> int:
    if isinstance(value, dict):
        return int(value["cost"])
    return int(value)


def build_queue() -> list[dict[str, int | str]]:
    if not FACTORY_DONE.exists():
        FACTORY_DONE.write_text("[]\n", encoding="utf-8")

    costs = _load_json(CURRENT_COSTS)
    task_hashes = _load_json(TASK_HASH_MAP)
    done = _done_tasks(_load_json(FACTORY_DONE))

    queue: list[dict[str, int | str]] = []
    for raw_task, raw_info in costs.items():
        task = _task_id(raw_task)
        cost = _cost(raw_info)
        if cost <= MIN_COST_EXCLUSIVE or task in done:
            continue
        hash_key = f"{task:03d}"
        task_hash = task_hashes.get(hash_key) or task_hashes.get(str(task))
        if not task_hash:
            raise KeyError(f"missing hash for task {task:03d}")
        queue.append({"task": task, "hash": str(task_hash), "cost": cost})

    queue.sort(key=lambda item: (-int(item["cost"]), int(item["task"])))
    return queue


def main() -> int:
    queue = build_queue()
    QUEUE_PATH.write_text(
        json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {QUEUE_PATH.relative_to(REPO_ROOT)} ({len(queue)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
