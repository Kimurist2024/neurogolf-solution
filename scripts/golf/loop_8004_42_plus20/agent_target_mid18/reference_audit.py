#!/usr/bin/env python3
"""Independent executable specifications for the four target generators.

This is deliberately model-free: it proves that the prose rules used by the
lane match every stored example and two independent fresh generator streams.
"""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (99, 279, 345, 239, 75, 392, 387, 225)
HASHES = {
    99: "444801d8",
    279: "b2862040",
    345: "d9f24cd1",
    239: "9af7a82c",
    75: "363442ee",
    392: "f8c80d96",
    387: "f35d900a",
    225: "93b581b8",
}
SEEDS = (91_018_000, 92_018_000)
FRESH_PER_SEED = 5_000


def solve_099(grid: list[list[int]]) -> list[list[int]]:
    """Fill each width-five blue frame with its sole non-blue seed color."""
    output = copy.deepcopy(grid)
    seeds = [
        (r, c, value)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value not in (0, 1)
    ]
    for seed_row, seed_col, color in seeds:
        left = seed_col - 2
        bottom = next(
            r
            for r in range(seed_row, len(grid))
            if grid[r][left : left + 5] == [1] * 5
        )
        top = bottom
        while top >= 0 and grid[top][left] == grid[top][left + 4] == 1:
            top -= 1
        for r in range(top, bottom + 1):
            for c in range(left, left + 5):
                if output[r][c] == 0:
                    output[r][c] = color
    return output


def solve_279(grid: list[list[int]]) -> list[list[int]]:
    """Recolor whole four-connected blue components iff they contain a cycle."""
    height, width = len(grid), len(grid[0])
    output = copy.deepcopy(grid)
    unseen = {(r, c) for r in range(height) for c in range(width) if grid[r][c] == 1}
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        component = {start}
        degree_sum = 0
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if grid[nr][nc] != 1:
                    continue
                degree_sum += 1
                if (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    component.add((nr, nc))
                    queue.append((nr, nc))
        edge_count = degree_sum // 2
        if edge_count >= len(component):
            for r, c in component:
                output[r][c] = 8
    return output


def solve_345(grid: list[list[int]]) -> list[list[int]]:
    """Run red paths upward, stepping right when gray blocks the next cell."""
    output = copy.deepcopy(grid)
    bottom = len(grid) - 1
    starts = [c for c, value in enumerate(grid[bottom]) if value == 2]
    for start in starts:
        r, c = bottom, start
        while r > 0:
            if output[r - 1][c] == 5:
                c += 1
            else:
                r -= 1
            output[r][c] = 2
    return output


def solve_239(grid: list[list[int]]) -> list[list[int]]:
    """Render descending, top-aligned frequency bars for all colors."""
    counts = Counter(value for row in grid for value in row)
    ordered = sorted(((count, color) for color, count in counts.items()), reverse=True)
    height = ordered[0][0]
    return [
        [color if r < count else 0 for count, color in ordered]
        for r in range(height)
    ]


def solve_075(grid: list[list[int]]) -> list[list[int]]:
    """Copy the left 3x3 tile into every block selected by a blue marker."""
    output = [[0 for _ in row] for row in grid]
    tile = [row[:3] for row in grid[:3]]
    for r in range(3):
        output[r][:3] = tile[r]
    for r in range(len(grid)):
        output[r][3] = 5
    markers = [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if c > 3 and value == 1
    ]
    for marker_row, marker_col in markers:
        top, left = marker_row - 1, marker_col - 1
        for dr in range(3):
            for dc in range(3):
                output[top + dr][left + dc] = tile[dr][dc]
    return output


def draw_392(
    row: int, col: int, thick: int, show: int, color: int
) -> tuple[list[list[int]], list[list[int]]]:
    grid = [[0] * 10 for _ in range(10)]
    output = [[5] * 10 for _ in range(10)]

    def draw(target: list[list[int]], r: int, c: int) -> None:
        if 0 <= r < 10 and 0 <= c < 10:
            target[r][c] = color

    for i in range(10):
        radius = (thick + 1) * i
        for r in range(row - radius + thick, row + radius):
            for c in (col - radius + thick, col + radius - 1):
                draw(output, r, c)
                if i <= show:
                    draw(grid, r, c)
        for c in range(col - radius + thick, col + radius):
            for r in (row - radius + thick, row + radius - 1):
                draw(output, r, c)
                if i <= show:
                    draw(grid, r, c)
    return grid, output


def solve_392(grid: list[list[int]]) -> list[list[int]]:
    """Invert the visible mat prefix and continue all clipped concentric mats."""
    color = next(value for row in grid for value in row if value != 0)
    centers = [(r, 0) for r in range(10)] + [(0, c) for c in range(1, 10)]
    for row, col in centers:
        for thick in (1, 2):
            for show in (2, 3):
                candidate, output = draw_392(row, col, thick, show, color)
                if candidate == grid:
                    return output
    raise ValueError("input is not a valid mat prefix")


def solve_387(grid: list[list[int]]) -> list[list[int]]:
    """Reconstruct the decorated rectangle from its four colored corners."""
    height, width = len(grid), len(grid[0])
    points = [(r, c) for r in range(height) for c in range(width) if grid[r][c]]
    top, bottom = min(r for r, _ in points), max(r for r, _ in points)
    left, right = min(c for _, c in points), max(c for _, c in points)
    colors = [grid[top][left], grid[top][right]]
    output = [[0] * width for _ in range(height)]
    corners = ((top, left, 0), (top, right, 1), (bottom, left, 1), (bottom, right, 0))
    for row, col, index in corners:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                output[row + dr][col + dc] = colors[1 - index]
        output[row][col] = colors[index]
    wide, tall = right - left, bottom - top
    for dc in range(2, 1 + wide // 2, 2):
        for row in (top, bottom):
            output[row][left + dc] = output[row][right - dc] = 5
    for dr in range(2, 1 + tall // 2, 2):
        for col in (left, right):
            output[top + dr][col] = output[bottom - dr][col] = 5
    return output


def solve_225(grid: list[list[int]]) -> list[list[int]]:
    """Copy the 2x2 source to the four clipped diagonal offset blocks."""
    size = len(grid)
    points = [(r, c) for r in range(size) for c in range(size) if grid[r][c]]
    row, col = min(r for r, _ in points), min(c for _, c in points)
    colors = [grid[row][col], grid[row][col + 1], grid[row + 1][col], grid[row + 1][col + 1]]
    output = copy.deepcopy(grid)
    for dr, dc, color in zip((2, 2, -2, -2), (2, -2, 2, -2), colors):
        for ddr in (0, 1):
            for ddc in (0, 1):
                rr, cc = row + dr + ddr, col + dc + ddc
                if 0 <= rr < size and 0 <= cc < size:
                    output[rr][cc] = color
    return output


SOLVERS: dict[int, Callable[[list[list[int]]], list[list[int]]]] = {
    99: solve_099,
    279: solve_279,
    345: solve_345,
    239: solve_239,
    75: solve_075,
    392: solve_392,
    387: solve_387,
    225: solve_225,
}


def normalize(grid: object) -> list[list[int]]:
    return [[int(value) for value in row] for row in grid]  # type: ignore[arg-type]


def check(solver: Callable[[list[list[int]]], list[list[int]]], example: dict) -> bool:
    return solver(normalize(example["input"])) == normalize(example["output"])


def main() -> None:
    sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
    sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
    results: dict[str, object] = {
        "purpose": "model-free verification of compiled generator rules",
        "fresh_per_seed": FRESH_PER_SEED,
        "seeds": list(SEEDS),
        "tasks": {},
    }
    for task in TASKS:
        solver = SOLVERS[task]
        generator = importlib.import_module(f"task_{HASHES[task]}")
        stored = json.loads(
            (ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text()
        )
        examples = stored["train"] + stored["test"] + stored["arc-gen"]
        known_right = sum(check(solver, example) for example in examples)
        streams = []
        for seed in SEEDS:
            random.seed(seed + task)
            right = errors = 0
            first_wrong = None
            for index in range(FRESH_PER_SEED):
                try:
                    example = generator.generate()
                    ok = check(solver, example)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    ok = False
                    if first_wrong is None:
                        first_wrong = {"index": index, "error": repr(exc)}
                right += int(ok)
                if not ok and first_wrong is None:
                    first_wrong = {
                        "index": index,
                        "input": example["input"],
                        "expected": example["output"],
                        "actual": solver(normalize(example["input"])),
                    }
            streams.append(
                {
                    "seed": seed + task,
                    "right": right,
                    "total": FRESH_PER_SEED,
                    "errors": errors,
                    "first_wrong": first_wrong,
                }
            )
        results["tasks"][str(task)] = {  # type: ignore[index]
            "hash": HASHES[task],
            "known": {"right": known_right, "total": len(examples)},
            "fresh": streams,
        }
        (HERE / "reference_audit.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        print(task, known_right, len(examples), streams, flush=True)


if __name__ == "__main__":
    main()
