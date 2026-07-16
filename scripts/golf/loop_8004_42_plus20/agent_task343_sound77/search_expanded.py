#!/usr/bin/env python3
"""Search exact 3-Conv / 2-comparison classifiers for task343.

The cost-172 budget allows exactly three scalar dynamic convolutions, two
boolean comparisons, and one boolean combiner.  Earlier searches covered only
row-aligned correlations.  This search adds column and point probes, including
the visibility-boundary probes used by the proven cost-178 compiler.
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
gen = importlib.import_module("task_d8c310e9")


def make_examples(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    req: list[int] = []
    for i in range(n):
        random.seed(seed + i)
        ex = gen.generate()
        x = np.asarray(ex["input"], dtype=np.int8)
        y = np.asarray(ex["output"], dtype=np.int8)
        ok6 = np.array_equal(x[:, np.arange(15) % 6], y)
        ok8 = np.array_equal(x[:, np.arange(15) % 8], y)
        xs.append(x)
        req.append(1 if ok6 and not ok8 else (-1 if ok8 and not ok6 else 0))
    return np.stack(xs), np.asarray(req, dtype=np.int8)


def feature(x: np.ndarray, spec: tuple[int, int, int, int]) -> np.ndarray:
    dh, top, dw, left = spec
    total = np.zeros(x.shape[0], dtype=np.int8)
    for r in range(5):
        rr = dh * r - top
        if not 0 <= rr < 5:
            continue
        for c in range(15):
            cc = dw * c - left
            if 0 <= cc < 15:
                total += x[:, r, c] == x[:, rr, cc]
    return total


def build_features(x: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    specs: list[tuple[int, int, int, int]] = []

    # All row-aligned affine correlations, including the negative-padding
    # shifts on which the exact 178 model relies.
    for dw in range(1, 31):
        for left in range(-30, (dw - 1) * 29 + 31):
            if any(0 <= dw * c - left < 15 for c in range(15)):
                specs.append((1, 0, dw, left))

    # A Conv with dw=30 selects one source/destination column pair and sums
    # equality over all five rows.  These expose the 3-vs-4 column motif.
    for c1 in range(15):
        for c2 in range(15):
            specs.append((1, 0, 30, 30 * c1 - c2))

    # Point probes against either top-row background anchor.  In particular,
    # (0,0)->(4,c) is an exact visibility boundary bit.
    for anchor_c in (0, 14):
        for r2 in range(5):
            for c2 in range(15):
                specs.append((30, 0, 30, 30 * anchor_c - c2 - r2 * 0))
                # top=-r2 maps source row zero to destination row r2.
                specs[-1] = (30, -r2, 30, 30 * anchor_c - c2)

    # All bottom-row point pairs capture color/silhouette equality without a
    # learned initializer and remain scalar in the official memory metric.
    for c1 in range(15):
        for c2 in range(15):
            specs.append((30, -4, 30, 30 * c1 - c2))

    unique: dict[bytes, tuple[tuple[int, int, int, int], np.ndarray]] = {}
    for spec in specs:
        f = feature(x, spec)
        unique.setdefault(f.tobytes(), (spec, f))
    return list(unique.values())


def relation(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    if op == "eq":
        return a == b
    if op == "gt":
        return a > b
    return a < b


def main() -> None:
    # A compact discovery set keeps the O(F^2*N) Boolean search interactive.
    # Any discovered expression is subsequently checked on independent large
    # samples and on the finite generator support before model adoption.
    x, req = make_examples(1_200, 343_800_517)
    only6 = req == 1
    only8 = req == -1
    feats = build_features(x)
    print(
        "examples", len(req), "features", len(feats),
        "only6", int(only6.sum()), "only8", int(only8.sum()),
        "both", int((req == 0).sum()), flush=True,
    )

    universal6: list[tuple[tuple[int, int], str, int]] = []
    zero8: list[tuple[tuple[int, int], str, int]] = []
    fm = np.stack([f for _, f in feats], axis=1)
    for i in range(len(feats)):
        ai = fm[:, i:i + 1]
        tail = fm[:, i + 1:]
        for op, pm in (("eq", ai == tail), ("gt", ai > tail), ("lt", ai < tail)):
            good6 = np.all(pm[only6], axis=0)
            good8 = ~np.any(pm[only8], axis=0)
            for off in np.flatnonzero(good6 | good8):
                j = i + 1 + int(off)
                p = pm[:, off]
                if good6[off] and good8[off]:
                    print("PERFECT", op, feats[i][0], feats[j][0])
                    return
                if good6[off]:
                    mask = int.from_bytes(np.packbits(p[only8]).tobytes())
                    universal6.append(((i, j), op, mask))
                if good8[off]:
                    # Store safe misses, so OR coverage is also a disjoint-mask
                    # query just like unsafe overlap for AND.
                    mask = int.from_bytes(np.packbits(~p[only6]).tobytes())
                    zero8.append(((i, j), op, mask))
    print("universal6", len(universal6), "zero8", len(zero8), flush=True)

    def find_disjoint(
        rels: list[tuple[tuple[int, int], str, int]], name: str,
    ) -> bool:
        # Two relation edges use <=3 Conv features iff they share a vertex.
        # Grouping by that vertex reduces the former quadratic global scan to
        # small groups, while Python integers make the bitset intersection
        # test a single machine-level operation.
        groups: list[list[int]] = [[] for _ in feats]
        for ri, rel in enumerate(rels):
            groups[rel[0][0]].append(ri)
            groups[rel[0][1]].append(ri)
        for group in groups:
            # Topologically different predicates with the same error mask are
            # interchangeable inside this shared-feature group.
            by_mask: dict[int, int] = {}
            for ri in group:
                by_mask.setdefault(rels[ri][2], ri)
            group = sorted(by_mask.values(), key=lambda ri: rels[ri][2].bit_count())
            for pos, ai in enumerate(group):
                a = rels[ai]
                for bi in group[pos + 1:]:
                    b = rels[bi]
                    if a[2] & b[2] == 0:
                        print("PERFECT", name)
                        for rel in (a, b):
                            print(rel[1], *(feats[k][0] for k in rel[0]))
                        return True
        return False

    # AND: unsafe pass masks must be disjoint. OR: safe miss masks must be
    # disjoint. Both relations already satisfy the opposite hard constraint.
    if find_disjoint(universal6, "AND"):
        return
    if find_disjoint(zero8, "OR"):
        return
    print("no exact sampled formula")


if __name__ == "__main__":
    main()
