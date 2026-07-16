#!/usr/bin/env python3
"""Verify input-only readable references for the three selected generators."""

from __future__ import annotations

import importlib
import json
import random
import sys
from collections import deque
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

TASKS = {156: "694f12f3", 237: "99fa7670", 345: "d9f24cd1"}


def ref156(grid: list[list[int]]) -> list[list[int]]:
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    boxes = []
    for row in range(height):
        for col in range(width):
            if grid[row][col] != 4 or (row, col) in seen:
                continue
            queue = deque([(row, col)])
            seen.add((row, col))
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and grid[nr][nc] == 4
                        and (nr, nc) not in seen
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            box = (min(rows), max(rows), min(cols), max(cols))
            area = (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
            boxes.append((area, box))
    if len(boxes) != 2 or boxes[0][0] == boxes[1][0]:
        raise ValueError(f"unexpected rectangle inventory: {boxes}")
    boxes.sort(key=lambda item: item[0])
    output = [row[:] for row in grid]
    for color, (_, (top, bottom, left, right)) in enumerate(boxes, 1):
        for row in range(top + 1, bottom):
            for col in range(left + 1, right):
                output[row][col] = color
    return output


def ref237(grid: list[list[int]]) -> list[list[int]]:
    height, width = len(grid), len(grid[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    for row in range(height):
        markers = [(col, color) for col, color in enumerate(grid[row]) if color]
        if not markers:
            continue
        if len(markers) != 1:
            raise ValueError(f"multiple markers in row {row}: {markers}")
        col, color = markers[0]
        for c in range(col, width):
            output[row][c] = color
        for r in range(row, height):
            output[r][width - 1] = color
    return output


def ref345(grid: list[list[int]]) -> list[list[int]]:
    size = len(grid)
    output = [row[:] for row in grid]
    starts = [col for col, color in enumerate(output[size - 1]) if color > 1]
    for start in starts:
        row, col = size - 1, start
        output[row][col] = 2
        while row > 0:
            if output[row - 1][col] == 5:
                col += 1
            else:
                row -= 1
            output[row][col] = 2
    return output


REFERENCES = {156: ref156, 237: ref237, 345: ref345}


def main() -> int:
    rows = []
    for task, task_hash in TASKS.items():
        module = importlib.import_module(f"task_{task_hash}")
        random.seed(8_003_400 + task)
        right = wrong = generation_errors = 0
        first_mismatch = None
        for index in range(5000):
            try:
                example = module.generate()
                actual = REFERENCES[task](example["input"])
                if actual == example["output"]:
                    right += 1
                else:
                    wrong += 1
                    if first_mismatch is None:
                        first_mismatch = index
            except Exception as exc:  # noqa: BLE001
                generation_errors += 1
                if first_mismatch is None:
                    first_mismatch = f"{index}: {type(exc).__name__}: {exc}"
        row = {
            "task": task,
            "generator_hash": task_hash,
            "fresh_requested": 5000,
            "right": right,
            "wrong": wrong,
            "generation_or_reference_errors": generation_errors,
            "first_mismatch": first_mismatch,
            "perfect": right == 5000 and wrong == 0 and generation_errors == 0,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    (HERE / "rule_reference_fresh5000.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if all(row["perfect"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
