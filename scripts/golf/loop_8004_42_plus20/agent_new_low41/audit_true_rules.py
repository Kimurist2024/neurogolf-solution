#!/usr/bin/env python3
"""Validate the compact Sakana rules against every known low41 pair."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (380, 242, 298, 26, 261, 351, 274, 317)

RULES = {
    380: "Rotate the fixed 3x3 grid 90 degrees clockwise.",
    242: "For each input row containing zero, reverse the prefix ending at its zero and keep its first three cells; discard rows without zero.",
    298: "Use row 2 as a three-color cyclic lookup: each input color selects the predecessor of its position in row 2.",
    26: "For every fixed 5x7 row, remove the center value and emit the first three values transformed by 8 >> (value + center).",
    261: "Move the last row to the front and reduce every color modulo 6.",
    351: "Flatten the 16x16 grid; locate color 3, then read five reversed length-5 rows at stride 16 immediately before it.",
    274: "Count input columns containing color 5; encode that count as a fixed 3x3 color-8 indicator pattern.",
    317: "On the fixed 9x9 grid, replicate each 3x3 source cell into a 3x3 block and output whether its color exceeds 4.",
}


def load_rule(task: int):
    path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"low41_task{task:03d}", path)
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
                    if [list(row) for row in actual] == expected:
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
