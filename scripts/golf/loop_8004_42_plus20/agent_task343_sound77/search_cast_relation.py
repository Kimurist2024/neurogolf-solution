#!/usr/bin/env python3
"""Search Cast(nonzero Conv) combined with one two-Conv relation.

Cast converts a scalar float Conv output directly to BOOL.  Consequently
Cast(A) AND/OR/XOR (B op C) uses three scalar Conv tensors and exactly three
boolean bytes, fitting the 15-byte classifier allowance at official cost 172.
"""

from __future__ import annotations

import numpy as np

from search_full_thresholds import build_features, examples


def main() -> None:
    x, req = examples(700, 343_172_999)
    hard = req != 0
    gold = req[hard] == 1
    fm, specs = build_features(x)
    index = {spec: i for i, spec in enumerate(specs)}

    def signature(spec: tuple[int, int, int, int]) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        dh, top, dw, left = spec
        vp = tuple((r, dh * r - top) for r in range(5) if 0 <= dh * r - top < 5)
        hp = tuple((c, dw * c - left) for c in range(15) if 0 <= dw * c - left < 15)
        return vp, hp

    canonical = {signature(spec): i for i, spec in enumerate(specs)}

    def resolve(spec: tuple[int, int, int, int]) -> int:
        return canonical[signature(spec)]

    # The proven 178 classifier's z11 point probe is nonzero for every true
    # 6-only case and directly identifies the visibility<=11 branch.
    ai = resolve((30, -4, 30, -11))
    cast_a = fm[:, ai] != 0

    wanted = [
        (1, 0, 1, 4), (1, 0, 1, 11), (30, -4, 30, -7),
        (1, 0, 11, 7), (1, 0, 12, 21), (1, 0, 15, 75),
    ]
    wanted += [(1, 0, 1, left) for left in range(-14, 15)]
    wanted += [(30, -r, 30, -c) for r in range(5) for c in range(15)]

    y = gold.astype(np.float32)
    hv = fm[hard].astype(np.float32)
    yc = y - y.mean()
    centered = hv - hv.mean(axis=0)
    denom = np.sqrt(np.sum(centered * centered, axis=0) * np.sum(yc * yc))
    corr = np.divide(
        np.abs(np.sum(centered * yc[:, None], axis=0)), denom,
        out=np.zeros(fm.shape[1], dtype=np.float32), where=denom != 0,
    )
    anchors = [resolve(s) for s in wanted]
    anchors.extend(int(i) for i in np.argsort(corr)[::-1][:200])
    anchors = list(dict.fromkeys(anchors))
    print(
        "examples", len(req), "features", fm.shape[1], "anchors", len(anchors),
        "cast", specs[ai], flush=True,
    )

    ah = cast_a[hard, None]
    for anum, bi in enumerate(anchors, 1):
        b = fm[:, bi:bi + 1]
        for op, rel in (("eq", b == fm), ("gt", b > fm), ("lt", b < fm)):
            rh = rel[hard]
            for comb, pred in (
                ("AND", ah & rh), ("OR", ah | rh), ("XOR", ah ^ rh),
            ):
                exact = np.all(pred == gold[:, None], axis=0)
                found = np.flatnonzero(exact)
                if found.size:
                    ci = int(found[0])
                    print("PERFECT", comb)
                    print("cast", specs[ai])
                    print(op, specs[bi], specs[ci])
                    return
        if anum % 10 == 0:
            print("checked", anum, specs[bi], flush=True)
    print("no exact sampled Cast+relation formula")


if __name__ == "__main__":
    main()
