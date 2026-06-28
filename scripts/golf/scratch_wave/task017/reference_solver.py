#!/usr/bin/env python3
"""Reference numpy solver for task017 (ARC 0dfd9992).

Rule: the output is a periodic background pattern
    value(row,col) = ((r*r + c*c) % mod) + 1,
    r = (offset+row) % length - length//2,   c = analogous for col.
The input is that same pattern with several rectangular cutouts painted
black (color 0). The task is to restore (fill) every black cutout cell
back to the underlying periodic pattern.

Period-independent reconstruction:
  * The pattern repeats every `length` (the generator guarantees
    length in [4,9]).  Any L in {4,...,9} that is consistent with the
    visible (non-black) cells is a valid period.  We pick the smallest.
  * With the true period L, copying each black cell from a neighbour at
    offset (0,+-L) / (+-L,0) brings the correct value.  Two sweeps of
    the four directions fill every <=5-wide/tall cutout (verified on
    fresh generator: 0 fails / 2000).
"""

from __future__ import annotations

import numpy as np

L_CANDIDATES = (4, 5, 6, 7, 8, 9)
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))
FILL_ITERS = 2


def _valid_period(grid: np.ndarray, L: int) -> bool:
    h, w = grid.shape
    a, b = grid[:, : w - L], grid[:, L:]
    m = (a != 0) & (b != 0)
    if np.any(a[m] != b[m]):
        return False
    a, b = grid[: h - L, :], grid[L:, :]
    m = (a != 0) & (b != 0)
    if np.any(a[m] != b[m]):
        return False
    return True


def _shift_copy(grid: np.ndarray, dr: int, dc: int, L: int) -> np.ndarray:
    h, w = grid.shape
    src = np.zeros_like(grid)
    r0, r1 = max(0, -dr * L), h - max(0, dr * L)
    c0, c1 = max(0, -dc * L), w - max(0, dc * L)
    src[r0:r1, c0:c1] = grid[r0 + dr * L : r1 + dr * L, c0 + dc * L : c1 + dc * L]
    take = (grid == 0) & (src != 0)
    out = grid.copy()
    out[take] = src[take]
    return out


def solve(grid: np.ndarray) -> np.ndarray:
    """grid is the 21x21 input label grid (0 = black cutout)."""
    L = next((c for c in L_CANDIDATES if _valid_period(grid, c)), L_CANDIDATES[0])
    out = grid.copy()
    for _ in range(FILL_ITERS):
        for dr, dc in DIRECTIONS:
            out = _shift_copy(out, dr, dc, L)
    return out


def _self_test() -> None:
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[4]
    data = json.load(open(root / "inputs/neurogolf-2026/task017.json"))
    fails = 0
    for key in ("train", "test", "arc-gen"):
        for pair in data[key]:
            inp = np.array(pair["input"])
            gold = np.array(pair["output"])
            if not np.array_equal(solve(inp), gold):
                fails += 1
    assert fails == 0, f"{fails} pairs failed"
    print("reference_solver: 0 fails on all train/test/arc-gen pairs")


if __name__ == "__main__":
    _self_test()
