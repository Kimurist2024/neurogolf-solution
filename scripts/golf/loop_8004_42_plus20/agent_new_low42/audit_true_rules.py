#!/usr/bin/env python3
"""Validate the compact Sakana rules against every known low42 pair."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (339, 126, 21, 171, 346, 227, 318, 332)

RULES = {
    339: "Flatten the 3x3 input, discard zero cells, and emit the remaining repeated color as one row.",
    126: "Keep all rows except the final empty row; on that final row place color 4 under each column belonging to a hollow monochrome 2x2 marker.",
    21: "Output a solid rectangle in the background color: its height and width are the minimum row/column color frequencies of the separator pattern.",
    171: "For an all-zero rectangular input, draw a one-cell-thick color-8 border.",
    346: "Return the nonzero color with the fewest adjacent equal pairs across rows and columns (the isolated/noise color).",
    227: "Compare the upper and lower 4x4 binary patterns cellwise; emit color 2 where they are equal and zero otherwise.",
    318: "Compare the upper and lower 4x4 patterns around the separator row; emit color 3 where either paired cell is nonzero.",
    332: "Scan each row with a parity offset based on row width; replace alternating color-5 cells by color 3 while leaving other cells unchanged.",
}


def load_rule(task: int):
    path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"low42_task{task:03d}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p


def main() -> None:
    rows = []
    for task in TARGETS:
        transform = load_rule(task)
        payload = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
        right = wrong = errors = 0
        split_counts: dict[str, int] = {}
        input_shapes: set[tuple[int, int]] = set()
        output_shapes: set[tuple[int, int]] = set()
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
                    actual = [list(row) for row in actual]
                    if actual == expected:
                        right += 1
                    else:
                        wrong += 1
                        if first_failure is None:
                            first_failure = {"split": split, "index": index, "kind": "wrong"}
                except Exception as exc:
                    errors += 1
                    if first_failure is None:
                        first_failure = {
                            "split": split,
                            "index": index,
                            "kind": "error",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
        total = right + wrong + errors
        rows.append({
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
        })
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
