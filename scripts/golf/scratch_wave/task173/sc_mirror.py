#!/usr/bin/env python3
"""Single-channel mirror: all spatial work on [30,30] label grids (no 10ch).

Uses Cauchy-Schwarz equality to detect uniform arm patterns per shape, avoiding
per-color one-hot convs. LUTs (length-10) are computed from full-copy anchors.
Translates to ONNX with only [1,1,30,30] spatial tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import correlate

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT / "inputs/neurogolf-2026/task173.json"

ARM_OFFSETS = {
    0: [(0, 0), (0, 2), (2, 0), (2, 2)],
    1: [(0, 1), (1, 0), (1, 2), (2, 1)],
    2: [(1, 0), (1, 2)],
    3: [(0, 1), (2, 1)],
}
NA = {s: len(ARM_OFFSETS[s]) for s in range(4)}


def armkern(s):
    k = np.zeros((3, 3), np.float32)
    for (r, c) in ARM_OFFSETS[s]:
        k[r, c] = 1.0
    return k


ARMK = {s: armkern(s) for s in range(4)}
BOX3 = np.ones((3, 3), np.float32)


def frame30(grid):
    H, W = grid.shape
    g = np.zeros((30, 30), np.int64)
    g[:H, :W] = grid
    return g


def C(mask, kern):
    return correlate(mask.astype(np.float32), kern, mode="constant", cval=0.0)


def solve30(grid):
    g = frame30(grid)
    L = g.astype(np.float32)
    nz = (g != 0).astype(np.float32)
    L2 = L * L
    nzbox = C(nz, BOX3)

    # Per shape: uniform arm detection (single channel).
    arm_uniform = {}
    arm_color = {}
    for s in range(4):
        nA = NA[s]
        cntNZ = C(nz, ARMK[s])
        sumL = C(L, ARMK[s])
        sumL2 = C(L2, ARMK[s])
        uniform = (np.abs(cntNZ - nA) < 0.5) & (np.abs(nA * sumL2 - sumL * sumL) < 0.5)
        arm_uniform[s] = uniform.astype(np.float32)
        # arm color = sumL/nA where uniform
        arm_color[s] = np.where(uniform, sumL / nA, 0.0)

    # Full-copy detection (center present, exactly nA+1 nonzero): learn LUTs.
    # c2s[a]=shape, c2p[a]=partner center color (for arm color a).
    c2s = -np.ones(10, np.int64)
    c2p = np.zeros(10, np.int64)
    for s in range(4):
        nA = NA[s]
        full = (arm_uniform[s] > 0.5) & (nz > 0.5) & (np.abs(nzbox - (nA + 1)) < 0.5)
        for (r, c) in np.argwhere(full):
            a = int(round(arm_color[s][r, c]))
            cc = int(g[r, c])
            if a == 0 or cc == 0 or a == cc:
                continue
            c2s[a] = s
            c2p[a] = cc
    # center LUT: cc2a[c]=arm color, cc2s[c]=shape (for center color c)
    cc2a = np.zeros(10, np.int64)
    cc2s = -np.ones(10, np.int64)
    for a in range(1, 10):
        if c2s[a] >= 0:
            cc2a[c2p[a]] = a
            cc2s[c2p[a]] = c2s[a]

    # ---- arm-pattern anchors (single channel) ----
    # For shape s, anchor where arm_uniform AND arm_color is an arm color of shape s
    # (i.e. c2s[arm_color]==s). Build per-shape, then combine.
    arm_anchor_mask = np.zeros((30, 30), np.float32)
    arm_a_grid = np.zeros((30, 30), np.int64)
    arm_s_grid = -np.ones((30, 30), np.int64)
    arm_cc_grid = np.zeros((30, 30), np.int64)
    for s in range(4):
        ac = arm_color[s]
        u = arm_uniform[s] > 0.5
        for (r, c) in np.argwhere(u):
            a = int(round(ac[r, c]))
            if a == 0 or c2s[a] != s:
                continue
            arm_anchor_mask[r, c] = 1.0
            arm_a_grid[r, c] = a
            arm_s_grid[r, c] = s
            arm_cc_grid[r, c] = c2p[a]

    # ---- center-based anchors (ms 0/1): pixel color is a center color ----
    cen_arm = np.array([cc2a[int(v)] for v in g.flatten()]).reshape(30, 30)
    cen_s = np.array([cc2s[int(v)] for v in g.flatten()]).reshape(30, 30)
    cen_cc = np.array([int(v) if cc2s[int(v)] >= 0 else 0 for v in g.flatten()]).reshape(30, 30)
    cen_mask = (cen_s >= 0)

    final_a = np.where(arm_s_grid >= 0, arm_a_grid, cen_arm)
    final_s = np.where(arm_s_grid >= 0, arm_s_grid, cen_s)
    final_cc = np.where(arm_s_grid >= 0, arm_cc_grid, cen_cc)
    anchor = (arm_s_grid >= 0) | cen_mask

    out = g.copy()
    for (r, c) in np.argwhere(anchor):
        s = int(final_s[r, c])
        if s < 0:
            continue
        a = int(final_a[r, c]); cc = int(final_cc[r, c])
        for (dr, dc) in ARM_OFFSETS[s]:
            rr, ccc = r + dr - 1, c + dc - 1
            if 0 <= rr < 30 and 0 <= ccc < 30:
                out[rr, ccc] = a
        out[r, c] = cc

    H, W = grid.shape
    return out[:H, :W]


def main():
    data = json.loads(TASK.read_text())
    total = bad = 0
    for subset in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data.get(subset, [])):
            grid = np.array(ex["input"], dtype=np.int64)
            gold = np.array(ex["output"], dtype=np.int64)
            pred = solve30(grid)
            total += 1
            if not np.array_equal(pred, gold):
                bad += 1
                if bad <= 4:
                    print(f"MISMATCH {subset}[{i}] dims={grid.shape} ndiff={int((pred!=gold).sum())}")
    print(f"total={total} bad={bad}")
    if bad == 0:
        print("OK: single-channel mirror exact on all pairs")


if __name__ == "__main__":
    main()
