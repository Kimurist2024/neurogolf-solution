"""Reference solver for task216 (ARC 8efcae92), archetype overlaps|random_pixels.

Rule (from generator task_8efcae92.py):
  - The grid contains 3-4 non-overlapping solid blue (color 1) rectangles
    ("boxes"), each with a 1-cell gap between them. Inside each box, some cells
    are red (color 2) -- these are random_pixels noise.
  - Exactly one box has the strictly-maximum number of red pixels (the
    generator guarantees unique red counts and that box idx 0 holds the max).
  - The OUTPUT is a crop of the winning box at its own (tall x wide)
    dimensions, blue background with the red pixels in place, placed at the
    top-left of the output grid.

This module reproduces the rule with numpy using only primitives that map
cleanly to static-shape ONNX ops (cumulative running-min / running-max for
run lengths, summed-area table for red counts). It asserts exactness on every
train + test pair before any ONNX build is attempted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT / "inputs" / "neurogolf-2026" / "task216.json"

BLUE = 1
RED = 2
SIZE = 30
BIG = 10_000  # sentinel larger than any coordinate


def decode(grid: list[list[int]]) -> np.ndarray:
    """Return a 30x30 int label grid (0 outside the actual grid)."""
    out = np.zeros((SIZE, SIZE), dtype=np.int64)
    g = np.array(grid, dtype=np.int64)
    out[: g.shape[0], : g.shape[1]] = g
    return out


def solve_grid(grid: list[list[int]]) -> np.ndarray:
    """Compute the winning-box crop, returned as a 30x30 label grid (output)."""
    lab = decode(grid)  # [30,30]
    nz = (lab > 0).astype(np.int64)  # box mask
    red = (lab == RED).astype(np.int64)

    # --- top-left corners: nz & up==0 & left==0 ---------------------------
    up = np.zeros_like(nz)
    up[1:, :] = nz[:-1, :]
    left = np.zeros_like(nz)
    left[:, 1:] = nz[:, :-1]
    corner = nz * (1 - up) * (1 - left)  # 1 at box top-left corners

    # --- run length to the right (within nz) ------------------------------
    # nearest zero column strictly to the right of each cell; for a corner the
    # box width = that column - j. Computed by a reverse running-min over the
    # column index of zero cells.
    cols = np.arange(SIZE)[None, :].repeat(SIZE, 0)  # [30,30] col index
    zero_col = np.where(nz == 0, cols, BIG)  # col idx where zero, else BIG
    # reverse running min along axis=1: nearest-zero-to-the-right column
    nz_right = np.minimum.accumulate(zero_col[:, ::-1], axis=1)[:, ::-1]
    width = nz_right - cols  # box width when evaluated at a corner

    # --- run length downward (within nz) ----------------------------------
    rows = np.arange(SIZE)[:, None].repeat(SIZE, 1)
    zero_row = np.where(nz == 0, rows, BIG)
    nz_down = np.minimum.accumulate(zero_row[::-1, :], axis=0)[::-1, :]
    height = nz_down - rows

    # --- summed-area table of reds ----------------------------------------
    sat = np.zeros((SIZE + 1, SIZE + 1), dtype=np.int64)
    sat[1:, 1:] = red.cumsum(0).cumsum(1)

    def rect_red(i: int, j: int, h: int, w: int) -> int:
        return int(sat[i + h, j + w] - sat[i, j + w] - sat[i + h, j] + sat[i, j])

    # --- pick winning corner by red count ---------------------------------
    best = None
    best_red = -1
    for i in range(SIZE):
        for j in range(SIZE):
            if corner[i, j]:
                h = int(height[i, j])
                w = int(width[i, j])
                rc = rect_red(i, j, h, w)
                if rc > best_red:
                    best_red = rc
                    best = (i, j, h, w)

    bi, bj, bh, bw = best
    out = np.zeros((SIZE, SIZE), dtype=np.int64)
    out[:bh, :bw] = lab[bi : bi + bh, bj : bj + bw]
    return out


def expected_grid(out: list[list[int]]) -> np.ndarray:
    arr = np.zeros((SIZE, SIZE), dtype=np.int64)
    g = np.array(out, dtype=np.int64)
    arr[: g.shape[0], : g.shape[1]] = g
    return arr


def main() -> None:
    data = json.loads(TASK.read_text())
    total = 0
    ok = 0
    for subset in ("train", "test"):
        for idx, pair in enumerate(data[subset]):
            pred = solve_grid(pair["input"])
            exp = expected_grid(pair["output"])
            match = pred.shape == exp.shape and bool((pred == exp).all())
            total += 1
            ok += match
            assert match, f"MISMATCH {subset}[{idx}]"
    print(f"reference_solver exact on {ok}/{total} visible pairs")


if __name__ == "__main__":
    main()
