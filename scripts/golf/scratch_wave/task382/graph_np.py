"""Numpy mirror: single-staircase + 1D-outside form for task382.

Derived from the verified build4 logic, restructured to compute ONE gather-shear
staircase (canon [20,20]) plus its transpose, selected by the horizontal/NT flag.
Outside sentinel stays as 1D broadcasts fed into Max. Cyan encoded as label 1.
"""
import numpy as np

P, Q, K = 30, 20, 5  # K=5 covers max staircase shift 4 (generator caps red rows at 4)


def graph(x):
    aQ = np.arange(Q)
    dd = np.arange(K)[:, None]
    sig = aQ[None, :]
    idxM = np.clip(sig - dd, 0, Q - 1)   # [K,20]
    idxP = np.clip(sig + dd, 0, Q - 1)   # [K,20]

    sel = np.zeros((3, 10), np.float32)
    sel[0, :] = 1.0
    sel[1, 2] = 1.0
    sel[2, 8] = 1.0
    ones = np.ones(P, np.float32)
    rowp = np.einsum('bchw,sc,w->bsh', x, sel, ones)[0][:, :Q]   # [3,20]
    colp = np.einsum('bchw,sc,h->bsw', x, sel, ones)[0][:, :Q]

    gpR = (rowp[0] > 0).astype(np.uint8)
    rpR = (rowp[1] > 0).astype(np.uint8)
    cpR = (rowp[2] > 0).astype(np.uint8)
    gpC = (colp[0] > 0).astype(np.uint8)
    rpC = (colp[1] > 0).astype(np.uint8)
    cpC = (colp[2] > 0).astype(np.uint8)

    NT = int(cpR.sum() == 1)   # horizontal case (no transpose)

    # Pre-select profiles into the canonical staircase frame.
    cum_prof = (np.where(NT, rpR, rpC)).astype(np.int32)   # red along staircase axis [20]
    cum_side = np.where(NT, cpR[0], cpC[0])                 # fwd vs rev cumsum
    seed = (np.where(NT, cpC, cpR)).astype(np.uint8)        # cyan seed vector [20]
    shear_side = np.where(NT, rpC[0], rpR[0])               # +shift vs -shift

    cumf = np.cumsum(cum_prof)
    cumr = np.cumsum(cum_prof[::-1])[::-1]
    shift = np.where(cum_side > 0, cumf, cumr)              # [20]

    srcv = (seed * 1).astype(np.uint8)                      # cyan encoded as 1
    pos = srcv[idxM]   # [K,20]
    neg = srcv[idxP]   # [K,20]
    bank = np.where(shear_side > 0, pos, neg)               # [K,20]
    canon = bank[shift, :]                                  # [20,20]
    canon_t = canon.T
    cyan_pattern = np.where(NT, canon, canon_t)             # [20,20]

    # red line value (2): outer-min of red row/col indicators.
    red_value = np.where(np.minimum(rpR[:, None], rpC[None, :]) > 0,
                         np.uint8(2), np.uint8(0))          # [20,20]

    row_out = np.where(gpR[:, None] > 0, np.uint8(0), np.uint8(255))   # [20,1]
    col_out = np.where(gpC[None, :] > 0, np.uint8(0), np.uint8(255))   # [1,20]

    color_idx = np.maximum.reduce([
        cyan_pattern, red_value,
        np.broadcast_to(row_out, (Q, Q)),
        np.broadcast_to(col_out, (Q, Q)),
    ])
    label = np.full((P, P), 255, np.uint8)
    label[:Q, :Q] = color_idx
    # channel_ids: channel c -> id c if c in {0,2,8} else 254; cyan label is 1 -> channel 8
    ch_ids = np.full(10, 254, np.uint8)
    ch_ids[0] = 0
    ch_ids[2] = 2
    ch_ids[8] = 1
    ch = ch_ids.reshape(1, 10, 1, 1)
    return (label[None, None] == ch).astype(np.float32)


def to_onehot(grid):
    g = np.asarray(grid)
    H, W = g.shape
    x = np.zeros((1, 10, P, P), np.float32)
    for r in range(H):
        for c in range(W):
            x[0, g[r, c], r, c] = 1.0
    return x


if __name__ == "__main__":
    import json
    d = json.load(open("inputs/neurogolf-2026/task382.json"))
    bad = 0
    tot = 0
    for split in ["train", "test", "arc-gen"]:
        for p in d[split]:
            tot += 1
            x = to_onehot(np.array(p["input"]))
            out = graph(x)
            g = np.array(p["output"])
            H, W = g.shape
            exp = np.zeros((1, 10, P, P), np.float32)
            for r in range(H):
                for c in range(W):
                    exp[0, g[r, c], r, c] = 1.0
            if not np.array_equal(out, exp):
                bad += 1
                if bad <= 3:
                    print("MISMATCH", split)
    print(f"graph_np mismatches {bad}/{tot}")
