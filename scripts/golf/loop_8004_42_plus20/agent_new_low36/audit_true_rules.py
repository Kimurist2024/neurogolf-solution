#!/usr/bin/env python3
"""Verify the compact Sakana transform against every known task pair."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (149, 390, 272, 147, 40, 176, 252, 127)
RULES = {
    149: "Split the 11x11 input into nine 3x3 panels separated by color 8; emit 1 exactly for panels whose cell-value sum exceeds 8.",
    390: "Move every color-5 object from inside the color-2 frame to the corresponding reflected position outside that frame.",
    272: "Change each isolated color-2 cell to color 1 while preserving color-2 cells with an orthogonal color-2 neighbor.",
    147: "Change color-3 cells to color 8 exactly when they have an orthogonally adjacent color-3 cell.",
    40: "Partition the fixed 10x10 grid into four 5x5 quadrants and recolor every nonzero cell with that quadrant's corner color.",
    176: "Apply the generator's fixed three-row, column-periodic bit mask, adding color 4 to the selected cells while preserving the input.",
    252: "In every row, recolor each nonzero cell at an odd column index to color 4; preserve all other cells.",
    127: "Decode each 3x3 panel's two marker colors into a uniform three-column color block, preserving color-5 separators.",
}


def load_rule(task: int):
    path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"low36_task{task:03d}", path)
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
        first_failure = None
        input_shapes: set[tuple[int, int]] = set()
        output_shapes: set[tuple[int, int]] = set()
        split_counts = {}
        for split in ("train", "test", "arc-gen"):
            examples = payload.get(split, [])
            split_counts[split] = len(examples)
            for index, example in enumerate(examples):
                source = copy.deepcopy(example["input"])
                expected = example["output"]
                input_shapes.add((len(source), len(source[0])))
                output_shapes.add((len(expected), len(expected[0])))
                try:
                    actual = transform(source)
                    normalized = [list(row) for row in actual]
                    if normalized == expected:
                        right += 1
                    else:
                        wrong += 1
                        if first_failure is None:
                            first_failure = {"split": split, "index": index}
                except Exception as exc:
                    errors += 1
                    if first_failure is None:
                        first_failure = {
                            "split": split,
                            "index": index,
                            "error": f"{type(exc).__name__}: {exc}",
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
                    "perfect": right == total,
                    "first_failure": first_failure,
                },
                "split_counts": split_counts,
                "input_shapes": [list(x) for x in sorted(input_shapes)],
                "output_shapes": [list(x) for x in sorted(output_shapes)],
            }
        )
        print(f"task{task:03d}: {right}/{total}", flush=True)
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
