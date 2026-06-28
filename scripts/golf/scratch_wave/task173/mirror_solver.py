#!/usr/bin/env python3
"""ONNX-shaped (conv-only) mirror of the task173 rule. Center-anchored.

Convention: a stamp's CENTER pixel is its anchor. 3x3 kernels are laid out in
the natural grid; scipy.correlate(mask, kern)[center] = weighted neighbourhood
sum read from input at (center + off - 1). ONNX Conv with pads=[1,1,1,1] does
the same correlation.

Shapes (natural 3x3, center at (1,1)):
  S0 X    arms (0,0)(0,2)(2,0)(2,2)
  S1 plus arms (0,1)(1,0)(1,2)(2,1)
  S2 horiz arms (1,0)(1,2)
  S3 vert arms (0,1)(2,1)
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


def arm_kernel(sid: int) -> np.ndarray:
    k = np.zeros((3, 3), dtype=np.float32)
    for (r, c) in ARM_OFFSETS[sid]:
        k[r, c] = 1.0
    return k


def to_onehot(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    X = np.zeros((10, H, W), dtype=np.float32)
    for k in range(10):
        X[k] = (grid == k)
    return X


def corr(mask: np.ndarray, kern: np.ndarray) -> np.ndarray:
    return correlate(mask.astype(np.float32), kern, mode="constant", cval=0.0)


def solve(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    X = to_onehot(grid)
    nz = (grid != 0).astype(np.float32)
    box3 = np.ones((3, 3), dtype=np.float32)
    cen_k = np.zeros((3, 3), dtype=np.float32); cen_k[1, 1] = 1.0
    armK = {s: arm_kernel(s) for s in range(4)}

    label = grid.astype(np.float32)

    arm_shape = np.zeros((10, 4), dtype=np.float32)
    arm_partner = np.zeros((10,), dtype=np.float32)

    nz_box = corr(nz, box3)  # total nonzero in 3x3 around each center
    cen_nz = corr(nz, cen_k)  # center nonzero (== nz itself, center-anchored)

    # Learn per (shape s, arm color k) from FULL copies. A full copy centered at
    # anchor a: all arms color k (armhit==nA), center nonzero & != k, and exactly
    # nA+1 nonzero cells in the 3x3 (no extras).
    for s in range(4):
        nA = len(ARM_OFFSETS[s])
        for k in range(1, 10):
            armhit = corr(X[k], armK[s])
            full = (armhit >= nA - 0.5) & (cen_nz > 0.5) & (np.abs(nz_box - (nA + 1)) < 0.5)
            for (ar, ac) in np.argwhere(full):
                cc = int(round(label[ar, ac]))  # center color (anchor IS center)
                if cc == 0 or cc == k:
                    continue
                arm_shape[k, s] = 1.0
                arm_partner[k] = cc

    out = grid.copy().astype(np.int64)

    for s in range(4):
        nA = len(ARM_OFFSETS[s])
        for k in range(1, 10):
            if arm_shape[k, s] < 0.5:
                continue
            cc = int(arm_partner[k])
            cen_mask = (grid == cc).astype(np.float32)
            cen_anchor = cen_mask > 0.5  # center of color cc present (ms 0/1)
            armhit = corr(X[k], armK[s])
            arm_anchor = armhit >= nA - 0.5  # full arm pattern present (ms 0/2)
            anchor = cen_anchor | arm_anchor  # boolean [H,W], anchor = center pos
            for (ar, ac) in np.argwhere(anchor):
                for (dr, dc) in ARM_OFFSETS[s]:
                    rr, ccc = ar + dr - 1, ac + dc - 1
                    if 0 <= rr < H and 0 <= ccc < W:
                        out[rr, ccc] = k
                out[ar, ac] = cc
    return out


def main() -> None:
    data = json.loads(TASK.read_text())
    total = bad = 0
    for subset in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data.get(subset, [])):
            grid = np.array(ex["input"], dtype=np.int64)
            gold = np.array(ex["output"], dtype=np.int64)
            pred = solve(grid)
            total += 1
            if not np.array_equal(pred, gold):
                bad += 1
                if bad <= 3:
                    print(f"MISMATCH {subset}[{i}] dims={grid.shape} ndiff={int((pred!=gold).sum())}")
    print(f"total={total} bad={bad}")
    if bad == 0:
        print("OK: mirror solver exact on all pairs")


if __name__ == "__main__":
    main()
