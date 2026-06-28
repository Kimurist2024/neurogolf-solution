"""Reference solver for task379 (ecdecbb3): cyan-line drip rendering.

Rule (from generator task_ecdecbb3.py draw()):
  Input already contains cyan (8) horizontal lines and red (2) seed pixels.
  For each red seed (row,col) and each cyan line:
    walk from row toward line; paint cells red along col until you either
    hit an existing cyan cell (break, no box) or reach the line (paint a
    3x3 cyan box centered there, center stays red).
  xpose only transposes the final grids; since input is post-xpose, we operate
  on the given grid directly (lines may be horizontal OR vertical after xpose).

We must reproduce output FROM input. Because xpose may make lines vertical,
the cleanest spec is: detect orientation. But generator always draws lines
HORIZONTAL pre-xpose; after xpose they are vertical. The input we receive is
already post-xpose. So the input can have either horizontal full-rows of cyan
(xpose=0) or full-columns of cyan (xpose=1). We handle both by replicating the
draw on the received grid in its native orientation.
"""
import json
import numpy as np

CYAN = 8
RED = 2


def solve(grid):
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()

    # Detect line orientation: a cyan "line" is a full row or full column of cyan.
    full_rows = [r for r in range(H) if np.all(g[r] == CYAN)]
    full_cols = [c for c in range(W) if np.all(g[:, c] == CYAN)]

    # Seeds = red pixels in input.
    seeds = list(zip(*np.where(g == RED)))

    if full_rows and not full_cols:
        # horizontal lines (xpose=0 style)
        lines = full_rows
        for (row, col) in seeds:
            for line in lines:
                dr = -1 if line < row else 1
                r = row
                while r != line:
                    if out[r][col] == CYAN:
                        break
                    out[r][col] = RED
                    r += dr
                if r == line:
                    for ddr in (-1, 0, 1):
                        for ddc in (-1, 0, 1):
                            rr, cc = r + ddr, col + ddc
                            if 0 <= rr < H and 0 <= cc < W:
                                out[rr][cc] = CYAN
                    out[r][col] = RED
    elif full_cols and not full_rows:
        # vertical lines (xpose=1 style) -> transpose logic on columns
        lines = full_cols
        for (row, col) in seeds:
            for line in lines:
                dc = -1 if line < col else 1
                c = col
                while c != line:
                    if out[row][c] == CYAN:
                        break
                    out[row][c] = RED
                    c += dc
                if c == line:
                    for ddr in (-1, 0, 1):
                        for ddc in (-1, 0, 1):
                            rr, cc = row + ddr, c + ddc
                            if 0 <= rr < H and 0 <= cc < W:
                                out[rr][cc] = CYAN
                    out[row][c] = RED
    else:
        # both present or neither: handle generic (shouldn't happen in this task)
        raise RuntimeError(f"ambiguous orientation rows={full_rows} cols={full_cols}")

    return out


def main():
    d = json.load(open('inputs/neurogolf-2026/task379.json'))
    total = 0
    bad = 0
    for split in ['train', 'test', 'arc-gen']:
        for idx, p in enumerate(d[split]):
            total += 1
            pred = solve(p['input'])
            gold = np.array(p['output'])
            if not np.array_equal(pred, gold):
                bad += 1
                if bad <= 5:
                    print(f"MISMATCH {split}[{idx}] shape{gold.shape}")
    print(f"total={total} bad={bad}")
    assert bad == 0, "reference solver not exact"
    print("ALL EXACT")


if __name__ == '__main__':
    main()
