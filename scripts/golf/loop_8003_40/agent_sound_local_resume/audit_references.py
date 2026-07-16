#!/usr/bin/env python3
"""Prove readable true-rule references against raw solvers and generators."""

from __future__ import annotations

import copy
import importlib
import json
import random
import runpy
import sys
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(TASKS))

Grid = list[list[int]]


def normalize(grid: object) -> Grid:
    return [list(row) for row in grid]  # type: ignore[arg-type]


def rule168(grid: Grid) -> Grid:
    """Extend the ray of every monochrome L triomino in a 2x2 box."""
    height, width = len(grid), len(grid[0])
    output = copy.deepcopy(grid)
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


def rule192(grid: Grid) -> Grid:
    """Exact raw rule: dynamic modal color plus horizontal/vertical cross test."""
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


def rule343(grid: Grid) -> Grid:
    """Repeat each row with period 6 or 8 as selected by the raw rule."""
    output = []
    for row in grid:
        period = 8 if row[:4] in (row[4:8], row[8:12]) else 6
        output.append((row[:period] * 3)[:15])
    return output


def rule344(grid: Grid) -> Grid:
    """Simultaneously rewrite orthogonally adjacent 2/3 pairs."""
    height, width = len(grid), len(grid[0])
    output = copy.deepcopy(grid)
    for row in range(height):
        for col in range(width):
            neighbors = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width:
                    neighbors.append(grid[rr][cc])
            if grid[row][col] == 2 and 3 in neighbors:
                output[row][col] = 0
            elif grid[row][col] == 3 and 2 in neighbors:
                output[row][col] = 8
    return output


def raw_solver(task: int) -> Callable[[Grid], Grid]:
    namespace = runpy.run_path(str(ROOT / "inputs" / "sakana-gcg-2025" / "raw" / f"task{task:03d}.py"))
    transform = namespace["p"]

    def wrapped(grid: Grid) -> Grid:
        return normalize(transform(copy.deepcopy(grid)))

    return wrapped


def main() -> None:
    specs = {
        168: (rule168, "task_6e19193c", 168_800_340),
        192: (rule192, "task_7e0986d6", 192_800_340),
        343: (rule343, "task_d8c310e9", 343_800_340),
        344: (rule344, "task_d90796e8", 344_800_340),
    }
    report: dict[str, object] = {}
    for task, (rule, module_name, seed) in specs.items():
        corpus = json.loads((ROOT / "inputs" / "neurogolf-2026" / f"task{task:03d}.json").read_text())
        raw = raw_solver(task)
        known_total = known_rule_right = raw_right = raw_rule_right = 0
        first_failure = None
        for subset in ("train", "test", "arc-gen"):
            for index, example in enumerate(corpus[subset]):
                known_total += 1
                expected = normalize(example["output"])
                readable = rule(normalize(example["input"]))
                raw_output = raw(normalize(example["input"]))
                known_rule_right += readable == expected
                raw_right += raw_output == expected
                raw_rule_right += raw_output == readable
                if first_failure is None and (readable != expected or raw_output != expected):
                    first_failure = {"subset": subset, "index": index}

        generator = importlib.import_module(module_name)
        random.seed(seed)
        fresh_total = fresh_right = generation_errors = 0
        for _ in range(5000):
            try:
                example = generator.generate()
            except Exception:
                generation_errors += 1
                continue
            fresh_total += 1
            fresh_right += rule(normalize(example["input"])) == normalize(example["output"])

        row = {
            "task": task,
            "raw_solver": f"inputs/sakana-gcg-2025/raw/task{task:03d}.py",
            "generator": f"inputs/arc-gen-repo/tasks/{module_name}.py",
            "known_total": known_total,
            "readable_rule_vs_gold": known_rule_right,
            "raw_solver_vs_gold": raw_right,
            "raw_solver_vs_readable_rule": raw_rule_right,
            "fresh_seed": seed,
            "fresh_total": fresh_total,
            "fresh_rule_vs_generator": fresh_right,
            "generation_errors": generation_errors,
            "first_failure": first_failure,
            "perfect": (
                known_rule_right == known_total
                and raw_right == known_total
                and raw_rule_right == known_total
                and fresh_total == 5000
                and fresh_right == 5000
                and generation_errors == 0
            ),
        }
        report[str(task)] = row
        print(json.dumps(row), flush=True)

    (HERE / "reference_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(row["perfect"] for row in report.values()):  # type: ignore[union-attr]
        raise SystemExit(1)


if __name__ == "__main__":
    main()
