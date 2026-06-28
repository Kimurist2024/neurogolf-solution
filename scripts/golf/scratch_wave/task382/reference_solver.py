"""Reference solver for task382 (f15e1fac).

Rule (canonical pre-flip/pre-gravity frame):
  - red (2) at (r,0) for r in rows
  - cyan (8) at (0,c) for c in cols
  - trail: inc starts 0; for r in 1..H-1: inc += 1 if r in rows.
           for each c in cols: if c+inc < W: out[r][c+inc]=cyan
  Then BOTH input and output go through flip_horiz (if flip) then apply_gravity(gravity).

INFERENCE from observed input only:
  We must reproduce output from the observed input grid alone (flip/gravity unknown).

Observation: apply_gravity is one of 4 dihedral-ish ops:
  g0: identity
  g1: transpose
  g2: vertical flip (rows reversed)
  g3: transpose then... -> actually g = reverse-rows if g>=2, then transpose if g%2==1.
  apply_gravity(grid,g): grid=grid[::-1] if g>=2; then transpose if g%2==1.

So the full forward map T = gravity(flip_horiz?(canonical)).
The input canonical has: red col on left edge, cyan dots on top row.
After flip_horiz, red col -> right edge, cyan stays top row (cols mirrored).
After gravity g, these edges move.

KEY INSIGHT for inference: The output is a deterministic function of the input grid
under the SAME forward transform. Equivalent strategy: invert the transform to recover
the canonical input, apply the canonical trail rule, then re-apply the transform.

But inverting requires knowing (flip, gravity). We can RECOVER (flip,gravity) from the
input by locating where the red line and cyan line are. However a fully general ONNX
that branches on orientation is heavy.

Simpler exact route used here for the reference (and to validate): brute force all
(flip in {0,1}, gravity in {0,1,2,3}); for each, invert input->canonical candidate, check
it has the canonical structure (red on left col, cyan on top row, rest blank),
build canonical output, forward-transform, and that's the answer. Because the generator
is deterministic and the structure is recognizable, exactly the right (flip,gravity)
reproduces it. We verify it matches gold.
"""
import json
import numpy as np

RED, CYAN = 2, 8


def apply_gravity(grid, gravity):
    g = grid[::-1] if gravity >= 2 else grid
    g = [list(r) for r in zip(*g)] if gravity % 2 == 1 else g
    return [list(r) for r in g]


def flip_horiz(grid):
    return [list(r[::-1]) for r in grid]


def inv_gravity(grid, gravity):
    """Inverse of apply_gravity for the given gravity."""
    g = grid
    # forward: rev-rows(if g>=2) then transpose(if g%2==1)
    # inverse: transpose(if g%2==1) then rev-rows(if g>=2)
    g = [list(r) for r in zip(*g)] if gravity % 2 == 1 else g
    g = g[::-1] if gravity >= 2 else g
    return [list(r) for r in g]


def build_canonical_output(H, W, rows_set, cols):
    out = [[0] * W for _ in range(H)]
    for r in rows_set:
        out[r][0] = RED
    for c in cols:
        out[0][c] = CYAN
    inc = 0
    for r in range(1, H):
        inc += 1 if r in rows_set else 0
        for c in cols:
            if c + inc >= W:
                continue
            out[r][c + inc] = CYAN
    return out


def recover_canonical_params(canon):
    """Given a candidate canonical INPUT grid, extract rows(red on col0), cols(cyan on row0).
    Returns (rows_set, cols) if structure is clean, else None."""
    H = len(canon)
    W = len(canon[0])
    rows_set = set()
    cols = []
    # red must be only on column 0
    for r in range(H):
        for c in range(W):
            v = canon[r][c]
            if v == 0:
                continue
            if v == RED:
                if c != 0:
                    return None
                rows_set.add(r)
            elif v == CYAN:
                if r != 0:
                    return None
                cols.append(c)
            else:
                return None
    cols.sort()
    return rows_set, cols


def solve(inp):
    inp = [list(r) for r in inp]
    H, W = len(inp), len(inp[0])
    for flip in (0, 1):
        for gravity in (0, 1, 2, 3):
            # invert gravity then invert flip to get canonical input candidate
            cand = inv_gravity(inp, gravity)
            if flip:
                cand = flip_horiz(cand)
            params = recover_canonical_params(cand)
            if params is None:
                continue
            rows_set, cols = params
            ch, cw = len(cand), len(cand[0])
            out_can = build_canonical_output(ch, cw, rows_set, cols)
            # forward
            o = flip_horiz(out_can) if flip else out_can
            o = apply_gravity(o, gravity)
            # must reconstruct the input's non-trail pixels exactly
            return o
    return None


if __name__ == "__main__":
    d = json.load(open("inputs/neurogolf-2026/task382.json"))
    total = 0
    bad = 0
    for split in ["train", "test", "arc-gen"]:
        for i, p in enumerate(d[split]):
            total += 1
            got = solve(p["input"])
            if got != p["output"]:
                bad += 1
                if bad <= 3:
                    print(f"MISMATCH {split}[{i}]")
    print(f"checked {total} pairs, mismatches={bad}")
    assert bad == 0, "reference solver does not match all pairs"
    print("OK: reference solver exact on all pairs")
