"""Prototype v2: per-cell radius from HORIZONTAL run length (robust to vertical clip).

green = union over k of dilate( exact_k, k ), where exact_k = maroon cells whose
horizontal run length == 2k. blue beam = per column, rows strictly below bottom maroon.
"""
import json
import numpy as np

R_MAX = 5


def hrun_eq(M: np.ndarray, w: int) -> np.ndarray:
    """Maroon cells whose horizontal maroon run has length exactly w.
    A cell is in a run of length>=w iff some length-w all-maroon horizontal window
    contains it. run==w iff (in a >=w run) and not (in a >=w+1 run)."""
    H, W = M.shape

    def cover_ge(width):
        # window all-maroon of given width, then mark all cells it covers
        if width > W:
            return np.zeros_like(M)
        # erosion: win[i,j]=1 if M[i, j..j+width-1] all 1
        win = M.copy()
        for d in range(1, width):
            shifted = np.zeros_like(M)
            shifted[:, :W - d] = M[:, d:]
            win = np.minimum(win, shifted)
        # win[i,j]=1 means run starting at j length>=width. cover all j..j+width-1
        cover = np.zeros_like(M)
        for d in range(0, width):
            shifted = np.zeros_like(M)
            shifted[:, d:] = win[:, :W - d]
            cover = np.maximum(cover, shifted)
        return cover & M

    ge_w = cover_ge(w)
    ge_w1 = cover_ge(w + 1)
    return ge_w & (1 - ge_w1)


def box_dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return mask
    H, W = mask.shape
    out = np.zeros_like(mask)
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            sr0 = max(0, -dr); sr1 = min(H, H - dr)
            dr0 = max(0, dr); dr1 = min(H, H + dr)
            sc0 = max(0, -dc); sc1 = min(W, W - dc)
            dc0 = max(0, dc); dc1 = min(W, W + dc)
            out[dr0:dr1, dc0:dc1] = np.maximum(
                out[dr0:dr1, dc0:dc1], mask[sr0:sr1, sc0:sc1])
    return out


def solve(inp: np.ndarray) -> np.ndarray:
    size = inp.shape[0]
    M = (inp == 9).astype(np.int32)
    green = np.zeros_like(M)
    for k in range(1, R_MAX + 1):
        exact = hrun_eq(M, 2 * k)
        if exact.sum() == 0:
            continue
        green = np.maximum(green, box_dilate(exact, k))
    # blue beam: per column rows strictly below bottom maroon
    cum = np.cumsum(M, axis=0)
    beam = np.zeros_like(M)
    beam[1:, :] = (cum[:-1, :] > 0).astype(np.int32)
    out = np.zeros_like(inp)
    out = np.where(beam == 1, 1, out)
    out = np.where(green == 1, 3, out)
    out = np.where(M == 1, 9, out)
    return out


if __name__ == "__main__":
    d = json.load(open("inputs/neurogolf-2026/task349.json"))
    total = bad = 0
    for split in ("train", "test", "arc-gen"):
        for p in d[split]:
            inp = np.array(p["input"], dtype=np.int32)
            gold = np.array(p["output"], dtype=np.int32)
            pred = solve(inp)
            total += 1
            if not np.array_equal(pred, gold):
                bad += 1
                if bad <= 3:
                    print(f"MISMATCH {split} shape={inp.shape} ndiff={(pred!=gold).sum()}")
    print(f"total={total} bad={bad}")
