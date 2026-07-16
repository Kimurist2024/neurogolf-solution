#!/usr/bin/env python3
"""Readable true-generator references for B26 tasks."""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
HASHES = {328: "d22278a0", 358: "e21d9049"}


def solve328(grid: list[list[int]]) -> list[list[int]]:
    size = len(grid)
    corners = [(row, col, grid[row][col]) for row in (0, size - 1) for col in (0, size - 1) if grid[row][col]]
    output = [[0] * size for _ in range(size)]
    for row in range(size):
        for col in range(size):
            ranked = sorted((abs(sr - row) + abs(sc - col), sr, sc, color) for sr, sc, color in corners)
            if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
                continue
            _, sr, sc, color = ranked[0]
            if max(abs(sr - row), abs(sc - col)) % 2 == 0:
                output[row][col] = color
    return output


def solve358(grid: list[list[int]]) -> list[list[int]]:
    height, width = len(grid), len(grid[0])
    row_counts = [sum(value != 0 for value in row) for row in grid]
    col_counts = [sum(grid[row][col] != 0 for row in range(height)) for col in range(width)]
    center_row = max(range(height), key=row_counts.__getitem__)
    center_col = max(range(width), key=col_counts.__getitem__)
    horizontal_positions = [col for col, value in enumerate(grid[center_row]) if value]
    vertical_positions = [row for row in range(height) if grid[row][center_col]]
    if len(horizontal_positions) not in (3, 4) or len(vertical_positions) != len(horizontal_positions):
        raise ValueError("generator cross invariant violated")
    h_start, v_start = min(horizontal_positions), min(vertical_positions)
    h_sequence = [grid[center_row][col] for col in range(h_start, h_start + len(horizontal_positions))]
    v_sequence = [grid[row][center_col] for row in range(v_start, v_start + len(vertical_positions))]
    output = [[0] * width for _ in range(height)]
    for col in range(width):
        output[center_row][col] = h_sequence[(col - h_start) % len(h_sequence)]
    for row in range(height):
        output[row][center_col] = v_sequence[(row - v_start) % len(v_sequence)]
    return output


def audit(task: int) -> dict[str, object]:
    solver = solve328 if task == 328 else solve358
    examples = json.loads((ROOT / "inputs/neurogolf-2026" / f"task{task:03d}.json").read_text())
    subsets: dict[str, dict[str, int]] = {}
    for subset in ("train", "test", "arc-gen"):
        right = sum(solver(pair["input"]) == pair["output"] for pair in examples[subset])
        subsets[subset] = {"right": right, "wrong": len(examples[subset]) - right}
    generator = importlib.import_module(f"task_{HASHES[task]}")
    random.seed(26_000_000 + task)
    fresh_right = fresh_wrong = errors = 0
    for _ in range(5000):
        try:
            pair = generator.generate()
            if solver(pair["input"]) == pair["output"]:
                fresh_right += 1
            else:
                fresh_wrong += 1
        except Exception:
            errors += 1
    return {
        "task": task,
        "generator_hash": HASHES[task],
        "known_subsets": subsets,
        "known_right": sum(row["right"] for row in subsets.values()),
        "known_wrong": sum(row["wrong"] for row in subsets.values()),
        "fresh_right": fresh_right,
        "fresh_wrong": fresh_wrong,
        "fresh_generation_errors": errors,
        "rule": (
            "unique nearest colored corner by Manhattan distance, gated by even Chebyshev distance"
            if task == 328
            else "extend the observed periodic horizontal and vertical cross sequences through the full grid"
        ),
    }


def main() -> int:
    rows = [audit(task) for task in (328, 358)]
    if any(row["known_wrong"] or row["fresh_wrong"] or row["fresh_generation_errors"] for row in rows):
        raise RuntimeError(rows)
    payload = {"tasks": rows}
    (HERE / "reference_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
