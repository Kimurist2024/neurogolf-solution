#!/usr/bin/env python3
"""Fully-tensor (no data-dependent python loops) mirror, ONNX-translatable.

All ops here map 1:1 to ONNX nodes operating on fixed [1,C,30,30] tensors:
  Conv (depthwise/grouped + full), Pad/Slice to 30x30, Cast, Mul/Add/Sub,
  Greater/Equal/And/Or (-> via min/max/relu on {0,1} masks), ReduceMax/Sum.

We canonicalize every grid into a 30x30 frame at the TOP-LEFT (matches the
official one-hot encoding: grid placed at [0:H,0:W], rest zero).

Pipeline:
  X[10,30,30] one-hot. nz = sum_{k>=1} X[k].
  For each shape s (4 fixed kernels) build armhit_s[10,30,30] = corr(X[k],armK_s)
    via grouped conv (same kernel across channels) -> actually per (s) one conv
    over all 10 channels (depthwise, shared kernel).
  Full-copy detection per (s,k):
    full_s_k = (armhit_s[k]==nA_s) & (nz_center>0) & (nzbox==nA_s+1)
  center color at anchor cc: we need, per (s,k), which center color. Encode by
    summing over channels: cen_color_onehot = X (the center pixel's channel).
  Learn arm_shape[10,4] and, crucially, for stamping we need per arm-color k its
    shape kernel and its center color channel. We compute these as small
    [10,...] tensors via reductions over the 30x30 anchor maps.
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
CENK = np.zeros((3, 3), np.float32); CENK[1, 1] = 1.0
BOX3 = np.ones((3, 3), np.float32)


def frame30(grid):
    H, W = grid.shape
    g = np.zeros((30, 30), np.int64)
    g[:H, :W] = grid
    return g


def onehot(g30):
    X = np.zeros((10, 30, 30), np.float32)
    for k in range(10):
        X[k] = (g30 == k)
    return X


def C(mask, kern):
    return correlate(mask.astype(np.float32), kern, mode="constant", cval=0.0)


def solve30(grid):
    g = frame30(grid)
    X = onehot(g)  # [10,30,30]
    nz = X[1:].sum(0)  # [30,30]
    nzbox = C(nz, BOX3)

    # Per shape s: armhit for every color channel (depthwise, shared kernel)
    armhit = {s: np.stack([C(X[k], ARMK[s]) for k in range(10)], 0) for s in range(4)}  # [10,30,30]

    # Per shape s, per color k: full-copy anchor mask (k is arm color)
    # nz center (anchor pixel nonzero)
    full = {}
    for s in range(4):
        nA = NA[s]
        hit = armhit[s]  # [10,30,30]
        arm_ok = (hit >= nA - 0.5).astype(np.float32)
        cen_ok = (nz >= 0.5).astype(np.float32)[None, :, :]
        box_ok = (np.abs(nzbox - (nA + 1)) < 0.5).astype(np.float32)[None, :, :]
        full[s] = arm_ok * cen_ok * box_ok  # [10,30,30]; 1 where k is arm of full copy s

    # arm_shape[k,s] = does color k ever act as arm of shape s? (max over grid)
    arm_shape = np.zeros((10, 4), np.float32)
    for s in range(4):
        arm_shape[:, s] = full[s].reshape(10, -1).max(1)
    # but must ensure center!=k and center!=0; full already requires center nonzero;
    # center==k impossible because armhit counts color-k arms and center is color k
    # only if grid center==k, but then it'd be counted... handle: exclude where
    # center channel == k. center color channel at anchor: argmax_c X[c] but
    # we need a tensor. We instead require nzbox==nA+1 AND arms==k AND center is a
    # DIFFERENT single color. If center were also k, armhit would be nA but the
    # center cell adds to X[k] not counted by arm kernel (center offset not in arm
    # kernel) -> center color could be k. Guard below using center-color channel.

    # center color channel per anchor: for shape s,color k full anchors, the center
    # color = the channel c (c>=1) with X[c] at the anchor pixel. Compute per-channel
    # center indicator: cen_is_c[c] = X[c] (anchor is center pixel). For a full
    # anchor of (s,k), partner center color cc satisfies cen_is_c[cc]=1.
    # Build arm_partner_onehot[k, c] aggregated over anchors and shapes.
    arm_partner = np.zeros((10, 10), np.float32)  # [armcolor k, centercolor c]
    for s in range(4):
        f = full[s]  # [10,30,30] over k
        for c in range(1, 10):
            # anchors where center color is c: f[k] * X[c]
            contrib = (f * X[c][None, :, :]).reshape(10, -1).max(1)  # [10] over k
            arm_partner[:, c] = np.maximum(arm_partner[:, c], contrib)
    # remove self (k==c)
    for k in range(10):
        arm_partner[k, k] = 0.0
    # Recompute arm_shape excluding self-center cases: arm_shape valid only if a
    # real partner exists.
    has_partner = (arm_partner.sum(1) > 0.5).astype(np.float32)  # [10]
    arm_shape = arm_shape * has_partner[:, None]

    # Now stamping.
    # Per arm color k: its shape kernel Kk = sum_s arm_shape[k,s]*ARMK[s].
    # Per arm color k: its center color channel = arm_partner[k] (one-hot length10).
    # Anchor map for type of arm-color k:
    #   cen_anchor_k = X[cc] (center color present)   -> sum_c arm_partner[k,c]*X[c]
    #   arm_anchor_k = (armhit_for_k_shape == nA_k)
    # anchor_k = max(cen_anchor_k, arm_anchor_k)  (boolean)
    # Then paint:
    #   arms: dilate anchor_k by Kk (transpose conv) -> arm color k at arm offsets
    #   center: at anchor positions, center color cc.

    # armhit for k's own shape: hk = sum_s arm_shape[k,s]*armhit[s][k]
    out = g.copy()

    nA_k = (arm_shape * np.array([NA[s] for s in range(4)])[None, :]).sum(1)  # [10]

    # Build per-color anchor map [10,30,30]
    anchor = np.zeros((10, 30, 30), np.float32)
    for k in range(1, 10):
        if arm_shape[k].sum() < 0.5:
            continue
        # k's shape index
        s = int(np.argmax(arm_shape[k]))
        cc_onehot = arm_partner[k]  # [10]
        cen_anchor = sum(cc_onehot[c] * X[c] for c in range(10))  # [30,30]
        arm_anchor = (armhit[s][k] >= nA_k[k] - 0.5).astype(np.float32)
        anchor[k] = np.maximum((cen_anchor >= 0.5).astype(np.float32), arm_anchor)

    # Paint arms: for each k, dilate anchor[k] by its shape kernel (place k at arm
    # offsets relative to center=anchor). Dilation by correlation with flipped
    # kernel == convolution; since arm kernels are symmetric except S? all 4 arm
    # kernels are symmetric under 180deg, so corr==conv. Use corr.
    arm_paint = np.zeros((10, 30, 30), np.float32)
    for k in range(1, 10):
        if arm_shape[k].sum() < 0.5:
            continue
        s = int(np.argmax(arm_shape[k]))
        arm_paint[k] = C(anchor[k], ARMK[s])  # arm color k spread to arm offsets
    arm_paint = (arm_paint >= 0.5).astype(np.float32)

    # Center paint: center color cc at anchor positions. For each arm color k with
    # partner cc, center anchor positions get color cc. Build center one-hot.
    cen_paint = np.zeros((10, 30, 30), np.float32)
    for k in range(1, 10):
        if arm_shape[k].sum() < 0.5:
            continue
        cc = int(np.argmax(arm_partner[k]))
        cen_paint[cc] = np.maximum(cen_paint[cc], anchor[k])

    # Compose output one-hot: start from input X, OR in arm_paint and cen_paint.
    outoh = X.copy()
    outoh[:] = np.maximum(outoh, arm_paint)
    for c in range(10):
        outoh[c] = np.maximum(outoh[c], cen_paint[c])
    # collapse to label (channels are mutually exclusive by construction since
    # non-overlap; argmax over channels, but channel 0 is background)
    # Build label: take max color channel that is 1.
    lab = np.zeros((30, 30), np.int64)
    for c in range(1, 10):
        lab = np.where(outoh[c] >= 0.5, c, lab)

    H, W = grid.shape
    return lab[:H, :W]


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
        print("OK: tensor solver exact on all pairs")


if __name__ == "__main__":
    main()
