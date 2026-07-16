#!/usr/bin/env python3
"""Verify the decoded compact rules against every known pair."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (320, 154, 393, 290, 336, 3, 58, 72)

RULES = {
    320: "For every red vertical bar, recolor its bottom floor(length/2) cells cyan.",
    154: "Reflect every gray outside-box pixel across its nearest red gripper boundary into the box, handling the transposed orientation.",
    393: "Count every nonzero color and emit the three colors in descending population order as a 3x1 column.",
    290: "Crop the nonzero square and swap its two nonzero colors everywhere.",
    336: "Fill the gray container interior cyan and extend cyan through its single opening to the corresponding grid edge.",
    3: "Continue the 2/3-row stencil from height 6 to height 9, preserving its optional alternating horizontal reflection, and recolor 1 to 2.",
    58: "Ignore the empty input content and draw the deterministic color-3 inward spiral for the observed square size.",
    72: "Split at the yellow row and emit color 3 exactly where the red top and bottom half masks differ.",
}


def load_rule(task: int):
    path = ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"low37_task{task:03d}", path)
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
