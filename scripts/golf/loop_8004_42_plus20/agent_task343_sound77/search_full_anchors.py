#!/usr/bin/env python3
"""Search cost-172 formulas sharing one informative affine Conv feature."""

from __future__ import annotations

import numpy as np

from search_full_thresholds import build_features, examples


def disjoint(
    rels: list[tuple[int, str, int]],
) -> tuple[tuple[int, str, int], tuple[int, str, int]] | None:
    by_mask: dict[int, tuple[int, str, int]] = {}
    for rel in rels:
        by_mask.setdefault(rel[2], rel)
    unique = sorted(by_mask.values(), key=lambda r: r[2].bit_count())
    for pos, a in enumerate(unique):
        for b in unique[pos + 1:]:
            if a[2] & b[2] == 0:
                return a, b
    return None


def main() -> None:
    x, req = examples(700, 343_517_172)
    only6 = req == 1
    only8 = req == -1
    fm, specs = build_features(x)
    spec_index = {spec: i for i, spec in enumerate(specs)}

    wanted: list[tuple[int, int, int, int]] = []
    # Proven classifier and incumbent approximate classifier features.
    wanted += [
        (1, 0, 1, 4), (1, 0, 1, 11), (30, -4, 30, -7),
        (30, -4, 30, -11), (1, 0, 11, 7), (1, 0, 12, 21),
        (1, 0, 15, 75),
    ]
    # All simple row-aligned shifts and visibility-boundary point probes.
    wanted += [(1, 0, 1, left) for left in range(-14, 15)]
    wanted += [(30, -r, 30, -c) for r in range(5) for c in range(15)]

    # Features with the largest standardized hard-label mean separation are
    # also plausible shared anchors for nonlinear pair comparisons.
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
    ranked = np.argsort(corr)[::-1][:100]

    anchors: list[int] = []
    for spec in wanted:
        if spec in spec_index:
            anchors.append(spec_index[spec])
    anchors.extend(int(i) for i in ranked)
    anchors = list(dict.fromkeys(anchors))
    print(
        "examples", len(req), "features", fm.shape[1], "anchors", len(anchors),
        "only6", int(only6.sum()), "only8", int(only8.sum()), flush=True,
    )

    for anum, ai in enumerate(anchors, 1):
        a = fm[:, ai:ai + 1]
        universal_by_mask: dict[int, tuple[int, str, int]] = {}
        zero8_by_mask: dict[int, tuple[int, str, int]] = {}
        for op, pm in (("eq", a == fm), ("gt", a > fm), ("lt", a < fm)):
            good6 = np.all(pm[only6], axis=0)
            good8 = ~np.any(pm[only8], axis=0)
            both = np.flatnonzero(good6 & good8)
            if both.size:
                bi = int(both[0])
                print("PERFECT SINGLE", op, specs[ai], specs[bi])
                return
            packed8 = np.packbits(pm[only8], axis=0)
            for bi in np.flatnonzero(good6):
                bi = int(bi)
                m = int.from_bytes(packed8[:, bi].tobytes())
                universal_by_mask.setdefault(m, (bi, op, m))
            packed6miss = np.packbits(~pm[only6], axis=0)
            for bi in np.flatnonzero(good8):
                bi = int(bi)
                m = int.from_bytes(packed6miss[:, bi].tobytes())
                zero8_by_mask.setdefault(m, (bi, op, m))
        universal = list(universal_by_mask.values())
        zero8 = list(zero8_by_mask.values())
        for name, rels in (("AND", universal), ("OR", zero8)):
            answer = disjoint(rels)
            if answer is not None:
                print("PERFECT", name, "anchor", specs[ai])
                for bi, op, _ in answer:
                    print(op, specs[ai], specs[bi])
                return
        if anum % 10 == 0:
            print("checked", anum, specs[ai], flush=True)
    print("no exact sampled shared-anchor formula")


if __name__ == "__main__":
    main()
