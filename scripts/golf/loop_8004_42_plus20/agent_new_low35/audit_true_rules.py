#!/usr/bin/env python3
"""Validate the eight compact Sakana rules against every known pair.

This is a rule audit only.  It does not create or promote ONNX candidates.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (50, 329, 350, 356, 371, 360, 214, 83)

RULES = {
    50: "Fill zero cells lying strictly between color-8 cells along either their row or column; preserve every other cell.",
    329: "In each odd-width row preserve only its center cell and replace all other cells with zero.",
    350: "Fill zero cells lying strictly between color-1 cells along either their row or column with color 8; preserve nonzero cells.",
    356: "At each cell, emit the bitwise union of the row-prefix/suffix maxima intersection and the column-prefix/suffix maxima intersection.",
    371: "Find the midpoint of the first two color-1 cells and paint the midpoint plus its four orthogonal neighbors color 3, preserving the input.",
    360: "For each row, fold the right four-cell arm onto the left four-cell arm with elementwise maximum, yielding width four.",
    214: "For the fixed 3x11 layout, keep the first four columns and append the reversed concatenation of the reversed-row first four cells and the current input column.",
    83: "Mirror the fixed 3x4 input horizontally and vertically with duplicated seams, yielding 6x8.",
}


def load_rule(task: int):
    path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"low35_task{task:03d}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p


def main() -> None:
    rows = []
    for task in TARGETS:
        transform = load_rule(task)
        payload = json.loads(
            (ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text()
        )
        right = wrong = errors = 0
        input_shapes: set[tuple[int, int]] = set()
        output_shapes: set[tuple[int, int]] = set()
        split_counts: dict[str, int] = {}
        first_failure = None
        for split, examples in payload.items():
            if not isinstance(examples, list):
                continue
            for index, example in enumerate(examples):
                if not isinstance(example, dict) or "input" not in example:
                    continue
                split_counts[split] = split_counts.get(split, 0) + 1
                grid = copy.deepcopy(example["input"])
                expected = example["output"]
                input_shapes.add((len(grid), len(grid[0])))
                output_shapes.add((len(expected), len(expected[0])))
                try:
                    actual = transform(grid)
                    if actual == expected:
                        right += 1
                    else:
                        wrong += 1
                        if first_failure is None:
                            first_failure = {"split": split, "index": index, "kind": "wrong"}
                except Exception as exc:  # fail closed
                    errors += 1
                    if first_failure is None:
                        first_failure = {
                            "split": split,
                            "index": index,
                            "kind": "error",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
        total = right + wrong + errors
        rows.append(
            {
                "task": task,
                "rule_summary": RULES[task],
                "known": {
                    "right": right,
                    "wrong": wrong,
                    "errors": errors,
                    "total": total,
                    "perfect": right == total and total > 0,
                    "first_failure": first_failure,
                },
                "split_counts": split_counts,
                "input_shapes": [list(shape) for shape in sorted(input_shapes)],
                "output_shapes": [list(shape) for shape in sorted(output_shapes)],
            }
        )
        print(f"task{task:03d}: rule={right}/{total} errors={errors}", flush=True)

    result = {
        "source": "inputs/sakana-gcg-2025/raw/taskNNN.py",
        "dataset": "inputs/neurogolf-2026/taskNNN.json",
        "targets_completed": len(rows),
        "all_perfect": all(row["known"]["perfect"] for row in rows),
        "rows": rows,
    }
    (HERE / "true_rule_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
