#!/usr/bin/env python3
"""Decode each generator's true transform and verify it on known/fresh cases."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Callable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
HASHES = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            queue = deque([(row, col)])
            seen[row, col] = True
            comp = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def solve051(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    coords = np.argwhere(grid != 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    counts = Counter(int(grid[r, c]) for r, c in coords)
    beam = min(counts, key=counts.get)
    er, ec = np.argwhere(grid == beam)[0]
    height, width = r1 - r0 + 1, c1 - c0 + 1
    if height < width:
        dr, dc = (1, 0) if er == r0 else (-1, 0)
        row, col = (r1 + 1, ec) if dr == 1 else (r0 - 1, ec)
    else:
        dr, dc = (0, 1) if ec == c0 else (0, -1)
        row, col = (er, c1 + 1) if dc == 1 else (er, c0 - 1)
    while 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
        out[row, col] = beam
        row, col = row + dr, col + dc
    return out


def solve064(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    background = Counter(grid.reshape(-1).tolist()).most_common(1)[0][0]
    candidates = []
    for color in set(grid.reshape(-1).tolist()) - {background}:
        coords = np.argwhere(grid == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            candidates.append(
                (area, color, (int(r0), int(r1), int(c0), int(c1)))
            )
    if not candidates:
        raise AssertionError("box not found")
    _, box_color, box = max(candidates)
    dot_colors = set(grid.reshape(-1).tolist()) - {background, box_color}
    if not dot_colors:
        return out
    dot = next(iter(dot_colors))
    r0, r1, c0, c1 = box
    for rr, cc in np.argwhere(grid == dot):
        rr, cc = int(rr), int(cc)
        if r0 <= rr <= r1 and cc < c0:
            dr, dc = 0, 1
        elif r0 <= rr <= r1 and cc > c1:
            dr, dc = 0, -1
        elif c0 <= cc <= c1 and rr < r0:
            dr, dc = 1, 0
        elif c0 <= cc <= c1 and rr > r1:
            dr, dc = -1, 0
        else:
            continue
        row, col = rr + dr, cc + dc
        while out[row, col] != box_color:
            out[row, col] = dot
            row, col = row + dr, col + dc
    return out


def solve185(grid: np.ndarray) -> np.ndarray:
    nonzero = grid[grid != 0]
    line = Counter(nonzero.tolist()).most_common(1)[0][0]
    if grid.shape[0] == 27:
        stride = 4
    else:
        candidates = []
        rows, cols = np.indices(grid.shape)
        for option in (3, 5):
            line_mask = ((rows + 1) % option == 0) | ((cols + 1) % option == 0)
            score = int(np.count_nonzero(grid[line_mask] == line)) - int(
                np.count_nonzero(grid[~line_mask] == line)
            )
            candidates.append((score, option))
        stride = max(candidates)[1]
    vertices: dict[tuple[int, int], int] = {}
    for row in range(stride - 1, grid.shape[0], stride):
        for col in range(stride - 1, grid.shape[1], stride):
            value = int(grid[row, col])
            if value not in (0, line):
                vertices[((row + 1) // stride, (col + 1) // stride)] = value
    if not vertices:
        raise AssertionError("colored vertices not found")
    min_row = min(row for row, _ in vertices)
    min_col = min(col for _, col in vertices)
    out = np.zeros((3, 3), dtype=grid.dtype)
    for row in range(3):
        for col in range(3):
            vals = [
                vertices.get((min_row + row + dr, min_col + col + dc), line)
                for dr in (0, 1)
                for dc in (0, 1)
            ]
            if len(set(vals)) == 1 and vals[0] != line:
                out[row, col] = vals[0]
    return out


def solve200(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid)
    row, col = np.argwhere(grid != 0)[0]
    color = int(grid[row, col])
    row, col = int(row), int(col)
    while col < grid.shape[1]:
        direction = -1 if row else 1
        out[row, col] = color
        while 0 <= row + direction < grid.shape[0]:
            row += direction
            out[row, col] = color
        col += 1
        if col >= grid.shape[1]:
            break
        out[row, col] = 5
        col += 1
    return out


def solve245(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    green = np.argwhere(grid == 3)
    red = np.argwhere(grid == 2)
    brow, bcol = green.min(axis=0)
    red_min = red.min(axis=0)
    target_r0, target_c0 = int(brow + 1), int(bcol + 1)
    dr = target_r0 - int(red_min[0])
    dc = target_c0 - int(red_min[1])
    out[grid == 2] = 0
    for row, col in red:
        out[int(row) + dr, int(col) + dc] = 2
    return out


DIGITS = {
    frozenset(((0, 0), (0, 1), (1, 0))): 0,
    frozenset(((0, 0), (0, 1), (0, 2), (1, 1))): 1,
    frozenset(((0, 1), (0, 2), (1, 2))): 2,
    frozenset(((0, 0), (1, 0), (1, 1), (2, 0))): 3,
    frozenset(): 4,
    frozenset(((0, 2), (1, 1), (1, 2), (2, 2))): 5,
    frozenset(((1, 0), (2, 0), (2, 1))): 6,
    frozenset(((1, 1), (2, 0), (2, 1), (2, 2))): 7,
    frozenset(((1, 2), (2, 1), (2, 2))): 8,
}


def solve264(grid: np.ndarray) -> np.ndarray:
    out = np.full((9, 9), 5, dtype=grid.dtype)
    block_origins = []
    for comp in components(grid != 0):
        cells = set(comp)
        options = []
        for r0 in range(min(row for row, _ in comp), max(row for row, _ in comp) - 1):
            for c0 in range(min(col for _, col in comp), max(col for _, col in comp) - 1):
                square = {(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)}
                if square <= cells:
                    options.append((r0, c0, square))

        def cover(remaining: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
            if not remaining:
                return []
            first = min(remaining)
            for r0, c0, square in options:
                if first in square and square <= remaining:
                    tail = cover(remaining - square)
                    if tail is not None:
                        return [(r0, c0), *tail]
            return None

        origins = cover(cells)
        if origins is None:
            raise AssertionError("sprite component is not exactly tiled by 3x3 blocks")
        block_origins.extend(origins)
    if len(block_origins) != 9:
        raise AssertionError(f"expected 9 sprites, got {len(block_origins)}")
    for r0, c0 in block_origins:
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        colored = frozenset(
            (row, col)
            for row in range(3)
            for col in range(3)
            if patch[row, col] != 5
        )
        digit = DIGITS[colored]
        if colored:
            color = int(next(patch[row, col] for row, col in colored))
            orow, ocol = 3 * (digit // 3), 3 * (digit % 3)
            for row, col in colored:
                out[orow + row, ocol + col] = color
    return out


def solve394(grid: np.ndarray) -> np.ndarray:
    black = np.argwhere(grid == 0)
    bite = int(round(math.sqrt(len(black))))
    row, col = black.min(axis=0)
    size = grid.shape[0]
    period = 2 if size < 7 else 3
    out = np.zeros((bite, bite), dtype=grid.dtype)
    for dr in range(bite):
        source_row = int(row) + dr
        source_row = (
            source_row + period
            if source_row + period < size
            else source_row - period
        )
        for dc in range(bite):
            out[dr, dc] = grid[source_row, int(col) + dc]
    return out


def solve397(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    boxes = components(grid != 0)
    for box in boxes:
        r0 = min(row for row, _ in box)
        c0 = min(col for _, col in box)
        colors = {int(grid[row, col]) for row, col in box}
        for row in range(r0 + 2, r0 + 2 + len(colors)):
            out[row, c0 : c0 + 2] = 3
    return out


SOLVERS: dict[int, Callable[[np.ndarray], np.ndarray]] = {
    51: solve051,
    64: solve064,
    185: solve185,
    200: solve200,
    245: solve245,
    264: solve264,
    394: solve394,
    397: solve397,
}


def verify_case(task: int, case: dict[str, object]) -> bool:
    actual = SOLVERS[task](np.asarray(case["input"], dtype=np.int64))
    expected = np.asarray(case["output"], dtype=np.int64)
    return np.array_equal(actual, expected)


def fresh_case(task: int, module: object) -> dict[str, object]:
    if task != 264:
        return module.generate()
    width, height = random.randint(14, 16), random.randint(14, 16)
    positions = [(r, c) for r in range(height - 2) for c in range(width - 2)]
    random.shuffle(positions)

    def compatible(left: tuple[int, int], right: tuple[int, int]) -> bool:
        lr, lc = left
        rr, rc = right
        return lr + 4 <= rr or rr + 4 <= lr or lc + 4 <= rc or rc + 4 <= lc

    def place(chosen: list[tuple[int, int]], available: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == 9:
            return chosen
        for index, pos in enumerate(available):
            if all(compatible(pos, old) for old in chosen):
                tail = [item for item in available[index + 1 :] if compatible(pos, item)]
                result = place([*chosen, pos], tail)
                if result is not None:
                    return result
        return None

    layout = place([], positions)
    if layout is None:
        raise AssertionError("constructive task264 layout search failed")
    random.shuffle(layout)
    rows = [row for row, _ in layout]
    cols = [col for _, col in layout]
    colors = [random.choice((1, 2, 3, 4, 6, 7, 8, 9)) for _ in range(9)]
    return module.generate(
        width=width,
        height=height,
        rows=rows,
        cols=cols,
        colors=colors,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=sorted(SOLVERS))
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seeds", default="80051620,80061620")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = args.task
    module = importlib.import_module(f"task_{HASHES[f'{task:03d}']}")
    known = module.validate()
    known_cases = [
        case for subset in ("train", "test") for case in known.get(subset, [])
    ]
    known_wrong = sum(not verify_case(task, case) for case in known_cases)
    rows = []
    for seed in [int(value) for value in args.seeds.split(",")]:
        random.seed(seed + task)
        wrong = 0
        first_error = None
        generated = 0
        for index in range(args.count):
            try:
                case = fresh_case(task, module)
                generated += 1
                if not verify_case(task, case):
                    wrong += 1
                    if first_error is None:
                        first_error = {"index": index, "input": case["input"], "output": case["output"]}
            except Exception as exc:
                wrong += 1
                if first_error is None:
                    first_error = {"index": index, "error": repr(exc)}
        rows.append(
            {
                "seed": seed,
                "generated": generated,
                "right": args.count - wrong,
                "wrong": wrong,
                "perfect": wrong == 0,
                "first_error": first_error,
            }
        )
    result = {
        "task": task,
        "hash": HASHES[f"{task:03d}"],
        "solver": SOLVERS[task].__name__,
        "known": {
            "count": len(known_cases),
            "right": len(known_cases) - known_wrong,
            "wrong": known_wrong,
            "perfect": known_wrong == 0,
        },
        "fresh": rows,
        "perfect": known_wrong == 0 and all(row["perfect"] for row in rows),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["perfect"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
