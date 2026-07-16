#!/usr/bin/env python3
"""Deduplicate all loose and harvested task232/task369 history."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.rank_dir import cost_of  # noqa: E402


TASKS = (232, 369)
SEARCH_ROOTS = ("artifacts", "inputs", "others", "scripts")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    checker_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as error:
        checker = False
        checker_error = f"{type(error).__name__}: {error}"
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        strict = True
    except Exception as error:
        strict = False
        strict_error = f"{type(error).__name__}: {error}"
    try:
        memory, params, cost = (int(value) for value in cost_of(str(path)))
        cost_error = None
    except Exception as error:
        memory = params = cost = None
        cost_error = f"{type(error).__name__}: {error}"
    return {
        "serialized_bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_params": sum(int(np.prod(item.dims)) if item.dims else 1 for item in model.graph.initializer),
        "max_einsum_inputs": max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0),
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_shape_inference": strict,
        "strict_error": strict_error,
        "memory": memory,
        "params": params,
        "cost": cost,
        "cost_error": cost_error,
    }


def main() -> int:
    by_task: dict[int, dict[str, list[str]]] = {task: defaultdict(list) for task in TASKS}
    seen = {task: 0 for task in TASKS}
    for root_name in SEARCH_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for task in TASKS:
            for path in root.rglob(f"task{task}*.onnx"):
                if HERE in path.parents:
                    continue
                seen[task] += 1
                by_task[task][digest(path)].append(str(path.relative_to(ROOT)))

    rows: list[dict[str, object]] = []
    for task in TASKS:
        for sha256, sources in sorted(by_task[task].items()):
            representative = ROOT / sorted(sources)[0]
            rows.append({"task": task, "sha256": sha256, "loose_sources": sorted(sources), **structural(representative)})

    harvest = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text())
    archive_rows = [row for row in harvest["rows"] if row.get("task") in TASKS]
    loose_keys = {(int(row["task"]), str(row["sha256"])) for row in rows}
    archive_only = [row for row in archive_rows if (int(row["task"]), str(row["sha256"])) not in loose_keys]
    payload = {
        "tasks": list(TASKS),
        "search_roots": list(SEARCH_ROOTS),
        "loose_files_seen": seen,
        "loose_distinct_hashes": {str(task): sum(row["task"] == task for row in rows) for task in TASKS},
        "loose_rows": rows,
        "harvest_archive_rows": archive_rows,
        "archive_only_rows": archive_only,
        "complete": True,
    }
    (HERE / "history_inventory.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"seen": seen, "distinct": payload["loose_distinct_hashes"], "harvest": len(archive_rows), "archive_only": len(archive_only)}, indent=2))
    for task in TASKS:
        print(f"task{task} below incumbent:")
        base = 116 if task == 232 else 130
        for row in sorted((r for r in rows if r["task"] == task and isinstance(r.get("cost"), int) and 0 <= r["cost"] < base), key=lambda r: r["cost"]):
            print(row["cost"], row["sha256"], row["loose_sources"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
