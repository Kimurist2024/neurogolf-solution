#!/usr/bin/env python3
"""Lean, label-grid-centric ONNX-translatable mirror (minimize 10ch tensors).

Idea: do learning to produce tiny per-color LUTs, then map the label grid
through the LUTs (Gather) to get per-pixel shape/partner grids, then stamp on
single-channel grids.

LUTs (length-10, indexed by color):
  c2s[color]  : shape id (0..3) if color is an ARM color, else -1
  c2p[color]  : partner CENTER color if color is an arm color, else 0
  cc2s[color] : shape id if color is a CENTER color, else -1
  cc2a[color] : partner ARM color if color is a center color, else 0

Stamping on label grids:
  arm anchors: for arm color a at pixel forming full arm pattern of c2s[a].
  center anchors: pixel whose color is a center color (any visible center).
  We compute an anchor LABEL grid `anchorArm` = arm color at center positions,
  and the shape there, then convolve to paint.

We avoid 10-channel spatial tensors: only single-channel [1,1,30,30] grids and
the one-hot input (given, free per scoring? no—input counts but is the model
input, excluded from memory). Intermediate 10ch tensors appear only briefly for
learning reductions; we keep them minimal.
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
    nzbox = C(nz, BOX3)

    # one-hot only used transiently for armhit per shape per color.
    X = np.stack([(g == k).astype(np.float32) for k in range(10)], 0)  # [10,30,30]

    # Learn LUTs. For each shape s and arm color a (1..9):
    #   full anchors: armhit_a_s==nA & center nonzero & nzbox==nA+1
    #   at those anchors center color = L (anchor is center pixel).
    c2s = -np.ones(10, np.int64)
    c2p = np.zeros(10, np.int64)
    cc2s = -np.ones(10, np.int64)
    cc2a = np.zeros(10, np.int64)
    for s in range(4):
        nA = NA[s]
        for a in range(1, 10):
            armhit = C(X[a], ARMK[s])
            full = (armhit >= nA - 0.5) & (nz > 0.5) & (np.abs(nzbox - (nA + 1)) < 0.5)
            for (r, c) in np.argwhere(full):
                cc = int(g[r, c])
                if cc == 0 or cc == a:
                    continue
                c2s[a] = s
                c2p[a] = cc
                cc2s[cc] = s
                cc2a[cc] = a

    # Stamp on label grid.
    out = g.copy()
    # Anchor arm grid: at each pixel, if it's a center of a stamp, the arm color &
    # shape come from cc2a/cc2s (center-based) OR from arm-pattern detection.
    # We paint in two label grids then OR with input.

    # (A) center-based anchors (ms 0/1): pixel color cc is a center color.
    #     arm color = cc2a[cc], shape = cc2s[cc]. Paint arms + keep center.
    # (B) arm-pattern anchors (ms 0/2): pixel is center of a full arm pattern of
    #     arm color a, shape c2s[a]. arm color a, center color = c2p[a].

    paint = np.zeros((30, 30), np.int64)

    # Build, per pixel, candidate (arm_color, shape, center_color) from center-based
    cen_arm = np.array([cc2a[int(v)] if v != 0 else 0 for v in g.flatten()]).reshape(30, 30)
    cen_shape = np.array([cc2s[int(v)] if v != 0 else -1 for v in g.flatten()]).reshape(30, 30)
    cen_cc = np.array([int(v) if cc2s[int(v)] >= 0 else 0 for v in g.flatten()]).reshape(30, 30)
    cen_anchor_mask = (cen_shape >= 0)

    # arm-pattern based: compute per shape armhit for the arm color present.
    # arm color at a center = c2p? no. We detect: for each shape s, anchors where
    # full arm pattern of SOME arm color a (with c2s[a]==s) exists.
    arm_anchor_a = np.zeros((30, 30), np.int64)   # arm color
    arm_anchor_s = -np.ones((30, 30), np.int64)   # shape
    arm_anchor_cc = np.zeros((30, 30), np.int64)  # center color
    for s in range(4):
        nA = NA[s]
        for a in range(1, 10):
            if c2s[a] != s:
                continue
            armhit = C(X[a], ARMK[s])
            hit = armhit >= nA - 0.5
            arm_anchor_a[hit] = a
            arm_anchor_s[hit] = s
            arm_anchor_cc[hit] = c2p[a]

    # Combine anchors: a pixel is an anchor if center-based OR arm-based.
    # arm color/shape/center color: prefer arm-based where present (it carries the
    # full arm color); center-based supplies for ms==1 (center only, no arms).
    final_a = np.where(arm_anchor_s >= 0, arm_anchor_a, cen_arm)
    final_s = np.where(arm_anchor_s >= 0, arm_anchor_s, cen_shape)
    final_cc = np.where(arm_anchor_s >= 0, arm_anchor_cc, cen_cc)
    anchor_mask = (arm_anchor_s >= 0) | cen_anchor_mask

    # Paint: for each anchor pixel, stamp arms (final_a) at final_s offsets, center final_cc.
    for (r, c) in np.argwhere(anchor_mask):
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
        print("OK: lean mirror exact on all pairs")


if __name__ == "__main__":
    main()
