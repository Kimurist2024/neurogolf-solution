#!/usr/bin/env python3
"""Readable generator-derived references for A2 targets."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GEN_DIR = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(GEN_DIR))

Grid = list[list[int]]


def rule009(grid: Grid) -> Grid:
    """Connect equal-colored endpoints on each line-grid row or column."""
    height, width = len(grid), len(grid[0])
    out = [[0] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            left = {grid[row][cc] for cc in range(col, -1, -3)}
            right = {grid[row][cc] for cc in range(col, width, 3)}
            up = {grid[rr][col] for rr in range(0, row, 3)}
            down = {grid[rr][col] for rr in range(row, height, 3)}
            colors = (left & right) | (up & down)
            out[row][col] = max(colors)
    return out


def rule077(grid: Grid) -> Grid:
    """Close each Chebyshev-radius-2 red component to its rectangle."""
    red = {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 2}
    unseen = set(red)
    closed: set[tuple[int, int]] = set()
    while unseen:
        component = {unseen.pop()}
        frontier = list(component)
        while frontier:
            r, c = frontier.pop()
            neighbors = {
                point
                for point in unseen
                if abs(point[0] - r) < 3 and abs(point[1] - c) < 3
            }
            unseen -= neighbors
            component |= neighbors
            frontier.extend(neighbors)
        r0 = min(r for r, _ in component)
        r1 = max(r for r, _ in component)
        c0 = min(c for _, c in component)
        c1 = max(c for _, c in component)
        closed |= {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    out = [row[:] for row in grid]
    for r, c in closed - red:
        out[r][c] = 4
    return out


def _plus(grid: Grid, row: int, col: int, radius: int) -> frozenset[tuple[int, int]]:
    # The raw program clips candidates at the grid boundary. Real crosses are
    # interior, but retaining clipped false candidates is necessary to mirror
    # its exact skip-first set-cover tie behavior.
    height, width = len(grid), len(grid[0])
    return frozenset(
        (rr, cc)
        for delta in range(-radius, radius + 1)
        for rr, cc in ((row + delta, col), (row, col + delta))
        if 0 <= rr < height and 0 <= cc < width
    )


def rule118(grid: Grid) -> Grid:
    """Find a disjoint plus cover of all red cells, then reveal gray crossings."""
    height, width = len(grid), len(grid[0])
    target = {(r, c) for r in range(height) for c in range(width) if grid[r][c] & 2}

    def cover(candidates: list[frozenset[tuple[int, int]]]) -> set[tuple[int, int]] | None:
        # Match the raw solver's skip-first recursion order exactly.
        def visit(index: int, used: frozenset[tuple[int, int]]):
            if target <= used:
                return set(used)
            if index == len(candidates):
                return None
            skipped = visit(index + 1, used)
            if skipped is not None:
                return skipped
            candidate = candidates[index]
            if candidate & used:
                return None
            return visit(index + 1, used | candidate)

        return visit(0, frozenset())

    for radius in (2, 3):
        candidates: list[frozenset[tuple[int, int]]] = []
        for row in range(height):
            for col in range(width):
                cells = _plus(grid, row, col, radius)
                if min(grid[r][c] for r, c in cells) & 2:
                    candidates.append(cells)
        selected = cover(candidates)
        if selected is None:
            continue
        out = [row[:] for row in grid]
        for row, col in selected:
            out[row][col] += 3 * (out[row][col] & 1)
        return out
    return [row[:] for row in grid]


def rule173(grid: Grid) -> Grid:
    """Copy complete 180-degree-symmetric 3x3 sprites into partial copies."""
    height, width = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    templates: list[list[int]] = []
    for row in range(height - 2):
        for col in range(width - 2):
            patch = [out[row + i // 3][col + i % 3] for i in range(9)]
            if patch == patch[::-1] and patch[4] and any(patch[:4]):
                templates.append(patch)
    for row in range(height - 2):
        for col in range(width - 2):
            patch = [out[row + i // 3][col + i % 3] for i in range(9)]
            for template in templates:
                if template[4] == patch[4] or sum(
                    template[i] == patch[i] for i in range(9)
                ) == 8:
                    for i, value in enumerate(template):
                        out[row + i // 3][col + i % 3] = value
                    patch = template
    return out


SPECS = {
    9: (rule009, "task_06df4c85", 900799913),
    77: (rule077, "task_36fdfd69", 7700799913),
    118: (rule118, "task_50846271", 11800799913),
    173: (rule173, "task_72322fa7", 17300799913),
}


def known(task: int, rule) -> tuple[int, int]:
    data = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    pairs = [pair for subset in ("train", "test", "arc-gen") for pair in data[subset]]
    return sum(rule(pair["input"]) == pair["output"] for pair in pairs), len(pairs)


def fresh(task: int, rule, module: str, seed: int, count: int) -> tuple[int, int]:
    generator = importlib.import_module(module)
    random.seed(seed)
    right = 0
    for _ in range(count):
        pair = generator.generate()
        right += rule(pair["input"]) == pair["output"]
    return right, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(SPECS))
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    tasks = [args.task] if args.task else sorted(SPECS)
    results = {}
    for task in tasks:
        rule, module, seed = SPECS[task]
        k = known(task, rule)
        f = fresh(task, rule, module, seed, args.count)
        results[str(task)] = {
            "known_right": k[0],
            "known_total": k[1],
            "fresh_right": f[0],
            "fresh_total": f[1],
            "generator": module,
            "seed": seed,
        }
        print(f"task{task:03d}: known={k[0]}/{k[1]} fresh={f[0]}/{f[1]}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
