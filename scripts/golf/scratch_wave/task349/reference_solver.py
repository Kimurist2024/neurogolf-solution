"""Reference numpy solver for task349 (db93a21d) 'death stars'.

INPUT: maroon (9) squares only. Each square is 2r x 2r, bottom row = r, left col = c.
OUTPUT (draw priority blue<green<maroon):
  - blue (1) beams: rows r+1..size-1, cols c..c+2r-1
  - green (3) halo: 4r x 4r square, rows r-3r+1..r+radius, cols c-radius..c+3r-1
  - maroon (9): same 2r x 2r square
We recover (r,c,radius) directly from each connected maroon square in the input.
"""
import json
import numpy as np


def solve(inp: np.ndarray) -> np.ndarray:
    size = inp.shape[0]
    out = np.zeros_like(inp)
    mask = (inp == 9).astype(np.int32)

    # Find connected maroon squares via labeling (squares never touch since
    # generator enforces non-overlap with separation). Recover bbox per square.
    seen = np.zeros_like(mask)
    stars = []
    for i in range(size):
        for j in range(size):
            if mask[i, j] and not seen[i, j]:
                # BFS flood
                stack = [(i, j)]
                seen[i, j] = 1
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < size and 0 <= nb < size and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = 1
                            stack.append((na, nb))
                rs = [a for a, _ in cells]
                cs = [b for _, b in cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                w = c1 - c0 + 1
                assert w % 2 == 0, f"odd width {w}"
                radius = w // 2
                # Square true rows = [r-w+1, r]. Recover r:
                #   if top clipped (r0==0): bottom r1 is intact -> r = r1
                #   else top intact -> r = r0 + w - 1 (handles bottom-clip & full)
                r = r1 if r0 == 0 else (r0 + w - 1)
                stars.append((r, c0, radius))

    # Draw in priority order: blue first, then green, then maroon
    for (r, c, radius) in stars:
        # blue beams below
        for row in range(r + 1, size):
            for dc in range(2 * radius):
                cc = c + dc
                if 0 <= cc < size:
                    out[row, cc] = 1
    for (r, c, radius) in stars:
        # green halo
        for dr in range(4 * radius):
            for dc in range(4 * radius):
                rr = r - dr + radius
                cc = c + dc - radius
                if 0 <= rr < size and 0 <= cc < size:
                    out[rr, cc] = 3
    for (r, c, radius) in stars:
        # maroon center
        for dr in range(2 * radius):
            for dc in range(2 * radius):
                rr = r - dr
                cc = c + dc
                if 0 <= rr < size and 0 <= cc < size:
                    out[rr, cc] = 9
    return out


if __name__ == "__main__":
    d = json.load(open("inputs/neurogolf-2026/task349.json"))
    total = 0
    bad = 0
    for split in ("train", "test", "arc-gen"):
        for p in d[split]:
            inp = np.array(p["input"], dtype=np.int32)
            gold = np.array(p["output"], dtype=np.int32)
            pred = solve(inp)
            total += 1
            if not np.array_equal(pred, gold):
                bad += 1
                if bad <= 3:
                    diff = np.argwhere(pred != gold)
                    print(f"MISMATCH {split} shape={inp.shape} ndiff={len(diff)}")
                    print("  first diffs:", diff[:5].tolist())
    print(f"total={total} bad={bad}")
    assert bad == 0, "solver not exact"
    print("REFERENCE SOLVER EXACT ON ALL PAIRS")
