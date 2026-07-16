#!/usr/bin/env python3
"""Search many Cast(A) AND/OR relation(B,C) cost-172 formulas."""

from __future__ import annotations

import numpy as np

from search_full_thresholds import build_features, examples


def packed_ints(bits: np.ndarray, columns: np.ndarray) -> list[tuple[int, int]]:
    packed = np.packbits(bits, axis=0)
    out: dict[int, int] = {}
    for col in columns:
        col = int(col)
        value = int.from_bytes(packed[:, col].tobytes())
        out.setdefault(value, col)
    return [(m, i) for m, i in out.items()]


def main() -> None:
    x, req = examples(700, 343_172_1001)
    only6 = req == 1
    only8 = req == -1
    fm, specs = build_features(x)
    truth = fm != 0

    auniv_cols = np.flatnonzero(np.all(truth[only6], axis=0))
    azero_cols = np.flatnonzero(~np.any(truth[only8], axis=0))
    # AND candidates are ranked by how few only8 examples Cast(A) passes.
    auniv = packed_ints(truth[only8], auniv_cols)
    auniv.sort(key=lambda z: z[0].bit_count())
    auniv = auniv[:300]
    # There are only a handful of distinct OR-side truth patterns.
    azero = packed_ints(~truth[only6], azero_cols)

    hard = req != 0
    y = (req[hard] == 1).astype(np.float32)
    hv = fm[hard].astype(np.float32)
    yc = y - y.mean()
    centered = hv - hv.mean(axis=0)
    denom = np.sqrt(np.sum(centered * centered, axis=0) * np.sum(yc * yc))
    corr = np.divide(
        np.abs(np.sum(centered * yc[:, None], axis=0)), denom,
        out=np.zeros(fm.shape[1], dtype=np.float32), where=denom != 0,
    )
    # Broad, deterministic anchor set: top correlation plus all exact/simple
    # horizontal shifts and a spread through the full affine family.
    anchors = list(map(int, np.argsort(corr)[::-1][:400]))
    anchors += list(range(0, fm.shape[1], max(1, fm.shape[1] // 200)))
    anchors = list(dict.fromkeys(anchors))
    print(
        "features", fm.shape[1], "anchors", len(anchors),
        "Cast-AND masks", len(auniv), "Cast-OR masks", len(azero), flush=True,
    )

    for anum, bi in enumerate(anchors, 1):
        b = fm[:, bi:bi + 1]
        for op, rel in (("eq", b == fm), ("gt", b > fm), ("lt", b < fm)):
            runiv_cols = np.flatnonzero(np.all(rel[only6], axis=0))
            rzero_cols = np.flatnonzero(~np.any(rel[only8], axis=0))
            runiv = packed_ints(rel[only8], runiv_cols)
            rzero = packed_ints(~rel[only6], rzero_cols)
            for am, ai in auniv:
                for rm, ci in runiv:
                    if am & rm == 0:
                        print("PERFECT AND")
                        print("cast", specs[ai])
                        print(op, specs[bi], specs[ci])
                        return
            for am, ai in azero:
                for rm, ci in rzero:
                    if am & rm == 0:
                        print("PERFECT OR")
                        print("cast", specs[ai])
                        print(op, specs[bi], specs[ci])
                        return
        if anum % 10 == 0:
            print("checked", anum, specs[bi], flush=True)
    print("no exact sampled multi-Cast formula")


if __name__ == "__main__":
    main()
