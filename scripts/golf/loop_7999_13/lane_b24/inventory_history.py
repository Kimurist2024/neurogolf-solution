#!/usr/bin/env python3
"""Deduplicate the complete loose/archive history for task256 and task257."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.rank_dir import cost_of  # noqa: E402


TASKS = (256, 257)
SEARCH_ROOTS = ("artifacts", "inputs", "others", "scripts")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    checker_error = None
    strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as error:  # fail-closed inventory evidence
        checker = False
        checker_error = f"{type(error).__name__}: {error}"
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        strict = True
    except Exception as error:  # fail-closed inventory evidence
        strict = False
        strict_error = f"{type(error).__name__}: {error}"
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    try:
        memory, params, cost = (int(value) for value in cost_of(str(path)))
        cost_error = None
    except Exception as error:  # static inventory must not imply eligibility
        memory = params = cost = None
        cost_error = f"{type(error).__name__}: {error}"
    return {
        "serialized_bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "max_einsum_inputs": max_einsum,
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
    paths_by_task_hash: dict[int, dict[str, list[str]]] = {
        task: defaultdict(list) for task in TASKS
    }
    files_seen = {task: 0 for task in TASKS}
    for root_name in SEARCH_ROOTS:
        search_root = ROOT / root_name
        if not search_root.exists():
            continue
        for task in TASKS:
            for path in search_root.rglob(f"task{task}*.onnx"):
                if HERE in path.parents:
                    continue
                files_seen[task] += 1
                paths_by_task_hash[task][digest(path)].append(str(path.relative_to(ROOT)))

    rows: list[dict[str, object]] = []
    for task in TASKS:
        for sha256, sources in sorted(paths_by_task_hash[task].items()):
            representative = ROOT / sorted(sources)[0]
            rows.append(
                {
                    "task": task,
                    "sha256": sha256,
                    "loose_sources": sorted(sources),
                    **structural(representative),
                }
            )

    harvest = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text()
    )
    archive_rows = [row for row in harvest["rows"] if row.get("task") in TASKS]
    present = {(int(row["task"]), str(row["sha256"])) for row in rows}
    archive_only = [
        row for row in archive_rows if (int(row["task"]), str(row["sha256"])) not in present
    ]

    payload = {
        "tasks": list(TASKS),
        "search_roots": list(SEARCH_ROOTS),
        "loose_files_seen": files_seen,
        "loose_distinct_hashes": {
            str(task): sum(row["task"] == task for row in rows) for task in TASKS
        },
        "loose_rows": rows,
        "harvest_archive_rows": archive_rows,
        "archive_only_rows": archive_only,
        "complete": True,
    }
    (HERE / "history_inventory.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "loose_files_seen": files_seen,
                "loose_distinct_hashes": payload["loose_distinct_hashes"],
                "harvest_rows": len(archive_rows),
                "archive_only": len(archive_only),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
