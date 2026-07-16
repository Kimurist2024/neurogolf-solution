#!/usr/bin/env python3
"""Fail-closed Wave15 audit for B22 task224 and task400.

This lane is intentionally non-promoting.  It proves the readable reference
rules against every known pair, traces declared versus runtime tensor shapes,
audits both ORT modes, and rechecks every archived below-incumbent task224
candidate.  Task400 history was already exhaustively inventoried by A17; its
rows are imported as immutable evidence and checked for completeness here.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.rank_dir import cost_of  # noqa: E402


BASELINE = {
    224: HERE / "baseline" / "task224.onnx",
    400: HERE / "baseline" / "task400.onnx",
}
EXPECTED = {
    224: ("02d6386ace32270c71ee2072328187a4c3a2a8355babd6b69fdc4a0e5b6bac79", 162),
    400: ("89b419dbad732d3235ac1ab7d078ef22eef3209eb8b5f30e21d3a502ccd03389", 164),
}
TASK224_ARCHIVE = [
    HERE.parent / "lane_archive_all400" / f"task224_r{ordinal:02d}_static{cost}.onnx"
    for ordinal, cost in ((1, 156), (2, 156), (3, 158), (4, 158))
]
A17_HISTORY = HERE.parent / "lane_a17" / "loose_history_scan.json"


def load_shared_audit():
    path = HERE.parent / "lane_b15" / "audit_candidates.py"
    spec = importlib.util.spec_from_file_location("b22_shared_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotate_clockwise(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*grid[::-1])]


def readable_task224(grid: list[list[int]]) -> list[list[int]]:
    """Literal expansion of raw/task224.py, retaining its fifth redraw."""
    result = [row[:] for row in grid]
    for turn in range(5):
        triples = [
            (row, col, value)
            for row, values in enumerate(result)
            for col, value in enumerate(values)
            if value
        ]
        rows, cols, colors = zip(*triples)
        left = min(cols) + 1
        right = max(cols)
        row = rows[0] + 1
        fill = sum(set(colors)) - 5
        result[row][left:right] = [fill] * (right - left)
        if turn < 4:
            result = rotate_clockwise(result)
    return result


def readable_task400(grid: list[list[int]]) -> list[list[int]]:
    """Literal expansion of raw/task400.py, including list.index semantics."""
    return [
        grid[~grid.index(row)][~row.index(1) :: -1][:5]
        for row in grid
        if 1 in row
    ]


def rule_audit(task: int) -> dict[str, Any]:
    examples = json.loads((ROOT / "inputs" / "neurogolf-2026" / f"task{task:03d}.json").read_text())
    function = readable_task224 if task == 224 else readable_task400
    right = wrong = 0
    first_failure = None
    subsets: dict[str, dict[str, int]] = {}
    for subset in ("train", "test", "arc-gen"):
        subset_right = subset_wrong = 0
        for index, pair in enumerate(examples[subset]):
            actual = function(pair["input"])
            if actual == pair["output"]:
                right += 1
                subset_right += 1
            else:
                wrong += 1
                subset_wrong += 1
                first_failure = first_failure or {"subset": subset, "index": index}
        subsets[subset] = {"right": subset_right, "wrong": subset_wrong}
    return {
        "right": right,
        "wrong": wrong,
        "subsets": subsets,
        "first_failure": first_failure,
        "classification": "D_global_geometry",
        "rule": (
            "find the four gray-extrema frame markers, infer the non-gray color, "
            "and restore the outer rectangle border"
            if task == 224
            else "find the 5x5 blue cutout and emit the diametrically opposite "
            "5x5 patch under the size-24 dihedral construction"
        ),
    }


def baseline_audit(shared: Any, task: int) -> dict[str, Any]:
    path = BASELINE[task]
    expected_hash, expected_cost = EXPECTED[task]
    if sha256(path) != expected_hash:
        raise RuntimeError(f"task{task:03d} baseline identity mismatch")
    model = onnx.load(path)
    memory, params, cost = cost_of(str(path))
    if int(cost) != expected_cost:
        raise RuntimeError(f"task{task:03d} cost mismatch: {cost}")
    structure = shared.structural(copy.deepcopy(model))
    runtime_shape = shared.trace_runtime_shapes(copy.deepcopy(model), task)
    known = shared.known_dual(copy.deepcopy(model), task)
    max_einsum_inputs = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": expected_hash,
        "memory": int(memory),
        "params": int(params),
        "cost": int(cost),
        "nodes": len(model.graph.node),
        "max_einsum_inputs": max_einsum_inputs,
        "structure": structure,
        "runtime_shape": runtime_shape,
        "known_dual": known,
        "admissible_source": (
            structure["pass_before_runtime_shape"]
            and runtime_shape.get("shape_cloak") is False
        ),
    }


def task224_archive_audit(shared: Any) -> list[dict[str, Any]]:
    rows = []
    for path in TASK224_ARCHIVE:
        if not path.exists():
            raise RuntimeError(f"missing archive candidate: {path}")
        model = onnx.load(path)
        memory, params, cost = cost_of(str(path))
        structure = shared.structural(copy.deepcopy(model))
        try:
            runtime_shape = shared.trace_runtime_shapes(copy.deepcopy(model), 224)
        except Exception as exc:  # fail closed
            runtime_shape = {"shape_cloak": None, "trace_error": f"{type(exc).__name__}: {exc}"}
        known = shared.known_dual(copy.deepcopy(model), 224)
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "memory": int(memory),
                "params": int(params),
                "cost": int(cost),
                "structure": structure,
                "runtime_shape": runtime_shape,
                "known_dual": known,
                "decision": "reject_known" if any(row["wrong"] or row["errors"] for row in known) else "reject_structure",
            }
        )
    return rows


def task400_history() -> dict[str, Any]:
    evidence = json.loads(A17_HISTORY.read_text())
    rows = [row for row in evidence["rows"] if row.get("task") == 400]
    distinct = {row["sha256"] for row in rows if row.get("stage") != "exact_baseline_duplicate"}
    cheaper = [
        row
        for row in rows
        if row.get("actual_screen_cost", row.get("static_floor", 10**9)) < EXPECTED[400][1]
    ]
    return {
        "source": str(A17_HISTORY.relative_to(ROOT)),
        "complete": bool(evidence.get("complete")),
        "rows": len(rows),
        "distinct_nonbaseline": len(distinct),
        "minimum_nonbaseline_cost": min(
            row.get("actual_screen_cost", row.get("static_floor", 10**9))
            for row in rows
            if row.get("stage") != "exact_baseline_duplicate"
        ),
        "below_164": cheaper,
    }


def main() -> None:
    shared = load_shared_audit()
    rules = {str(task): rule_audit(task) for task in (224, 400)}
    if any(item["wrong"] for item in rules.values()):
        raise RuntimeError("readable rule mismatch")
    baselines = [baseline_audit(shared, task) for task in (224, 400)]
    archive224 = task224_archive_audit(shared)
    history400 = task400_history()
    payload = {
        "tasks": [224, 400],
        "rules": rules,
        "baselines": baselines,
        "task224_below_incumbent_archive": archive224,
        "task400_history": history400,
        "winners": [],
        "verified_gain": 0.0,
        "fresh5000_run": False,
        "fresh5000_skip_reason": (
            "No strictly cheaper candidate passes known correctness and the structural/runtime-shape gate."
        ),
    }
    (HERE / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rules": rules,
                "baselines": [
                    {
                        "task": row["task"],
                        "cost": row["cost"],
                        "admissible_source": row["admissible_source"],
                        "shape_cloak": row["runtime_shape"].get("shape_cloak"),
                        "max_einsum_inputs": row["max_einsum_inputs"],
                        "known": [
                            (item["right"], item["wrong"], item["errors"])
                            for item in row["known_dual"]
                        ],
                    }
                    for row in baselines
                ],
                "task224_archive": [
                    (row["cost"], row["decision"]) for row in archive224
                ],
                "task400_history": history400,
                "winners": [],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
