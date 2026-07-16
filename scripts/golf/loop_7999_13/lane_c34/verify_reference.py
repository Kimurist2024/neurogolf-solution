#!/usr/bin/env python3
"""Verify a readable task009 reference against every repository JSON pair."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402


def reference(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = []
    for row in range(rows):
        out_row = []
        for col in range(cols):
            horizontal_left = set(grid[row][col::-3])
            horizontal_right = set(grid[row][col::3])
            column = [grid[r][col] for r in range(rows)]
            vertical_up = set(column[:row:3])
            vertical_down = set(column[row::3])
            common = (horizontal_left & horizontal_right) | (vertical_up & vertical_down)
            out_row.append(max(common))
        result.append(out_row)
    return result


def main() -> int:
    examples = scoring.load_examples(9)
    right = wrong = 0
    first_failure = None
    by_split = {}
    for split in ("train", "test", "arc-gen"):
        split_right = split_wrong = 0
        for index, example in enumerate(examples[split]):
            actual = reference(example["input"])
            if actual == example["output"]:
                right += 1
                split_right += 1
            else:
                wrong += 1
                split_wrong += 1
                first_failure = first_failure or {"split": split, "index": index}
        by_split[split] = {"right": split_right, "wrong": split_wrong}
    result = {
        "task": 9,
        "raw_rule": "max(horizontal step3 both-side intersection union vertical step3 up/down intersection)",
        "by_split": by_split,
        "right": right,
        "wrong": wrong,
        "first_failure": first_failure,
        "passed": wrong == 0,
    }
    (HERE / "reference_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if wrong == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
