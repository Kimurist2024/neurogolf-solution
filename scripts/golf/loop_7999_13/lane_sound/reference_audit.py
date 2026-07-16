#!/usr/bin/env python3
"""Readable true-rule references for tasks 192, 344 and 168.

The references are derived from the raw solvers and generator sources.  They
are intentionally ordinary grid code: no examples, coordinates, colors, or
output tables are embedded.
"""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(TASKS))

Grid = list[list[int]]


def task192_rule(grid: Grid) -> Grid:
    counts = [sum(row.count(color) for row in grid) for color in range(10)]
    selected = max(range(1, 10), key=lambda color: counts[color])
    height, width = len(grid), len(grid[0])
    output = [[0] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            horizontal = any(
                grid[row][other] == selected
                for other in range(max(0, col - 1), min(width, col + 2))
            )
            vertical = any(
                grid[other][col] == selected
                for other in range(max(0, row - 1), min(height, row + 2))
            )
            if grid[row][col] != 0 and horizontal and vertical:
                output[row][col] = selected
    return output


def task344_rule(grid: Grid) -> Grid:
    height, width = len(grid), len(grid[0])
    adjacent_23: set[tuple[int, int]] = set()
    adjacent_32: set[tuple[int, int]] = set()
    for row in range(height):
        for col in range(width):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if not (0 <= rr < height and 0 <= cc < width):
                    continue
                if grid[row][col] == 2 and grid[rr][cc] == 3:
                    adjacent_23.add((row, col))
                if grid[row][col] == 3 and grid[rr][cc] == 2:
                    adjacent_32.add((row, col))
    output = [row[:] for row in grid]
    for row, col in adjacent_23:
        output[row][col] = 0
    for row, col in adjacent_32:
        output[row][col] = 8
    return output


def task168_rule(grid: Grid) -> Grid:
    """Detect every monochrome L triomino and extend its missing-corner ray."""
    height, width = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    for top in range(height - 1):
        for left in range(width - 1):
            cells = [
                grid[top][left],
                grid[top][left + 1],
                grid[top + 1][left],
                grid[top + 1][left + 1],
            ]
            colors = {value for value in cells if value != 0}
            if len(colors) != 1 or cells.count(0) != 1:
                continue
            color = next(iter(colors))
            if cells.count(color) != 3:
                continue
            missing = cells.index(0)
            missing_row = top + missing // 2
            missing_col = left + missing % 2
            dr = -1 if missing // 2 == 0 else 1
            dc = -1 if missing % 2 == 0 else 1
            row, col = missing_row + dr, missing_col + dc
            while 0 <= row < height and 0 <= col < width:
                output[row][col] = color
                row += dr
                col += dc
    return output


def known_check(task: int, rule: Callable[[Grid], Grid]) -> tuple[int, int]:
    data = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    right = total = 0
    for subset in ("train", "test", "arc-gen"):
        for example in data[subset]:
            total += 1
            right += rule(example["input"]) == example["output"]
    return right, total


def fresh_check(module_name: str, rule: Callable[[Grid], Grid], count: int, seed: int) -> tuple[int, int]:
    generator = importlib.import_module(module_name)
    random.seed(seed)
    right = 0
    for _ in range(count):
        example = generator.generate()
        right += rule(example["input"]) == example["output"]
    return right, count


def main() -> None:
    specs = {
        192: (task192_rule, "task_7e0986d6", 192799913),
        344: (task344_rule, "task_d90796e8", 344799913),
        168: (task168_rule, "task_6e19193c", 168799913),
    }
    result: dict[str, object] = {}
    for task, (rule, module, seed) in specs.items():
        known = known_check(task, rule)
        fresh = fresh_check(module, rule, 5000, seed)
        result[str(task)] = {
            "known_right": known[0],
            "known_total": known[1],
            "fresh_right": fresh[0],
            "fresh_total": fresh[1],
            "seed": seed,
            "generator": module,
        }
        print(f"task{task:03d} known={known[0]}/{known[1]} fresh={fresh[0]}/{fresh[1]}")
    (HERE / "reference_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
