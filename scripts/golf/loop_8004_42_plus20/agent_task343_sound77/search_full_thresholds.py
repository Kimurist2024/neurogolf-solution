#!/usr/bin/env python3
"""Search every affine scalar Conv against reusable constant Conv probes.

There are only 53 distinct vertical and 1,058 distinct horizontal affine
pairings on the real 5x15 support.  Their Cartesian product exhausts all
scalar dynamic-Conv features obtainable through dilation/padding.  Comparing
two features with a shared constant Conv fits the strict three-Conv cost-172
budget.
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


def affine_maps(size: int) -> list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    out: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}
    for dilation in range(1, 31):
        for source in range(size):
            for dest in range(size):
                pad = dilation * source - dest
                pairs = tuple(
                    (i, dilation * i - pad)
                    for i in range(size)
                    if 0 <= dilation * i - pad < size
                )
                out.setdefault(pairs, (dilation, pad))
    return list(out.items())


def examples(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
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


def build_features(x: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    vmaps = affine_maps(5)
    hmaps = affine_maps(15)
    blocks: list[np.ndarray] = []
    specs: list[tuple[int, int, int, int]] = []
    for vpairs, (dh, top) in vmaps:
        # Collapse the selected row-pair equalities for every column pair.
        col_pair = np.zeros((x.shape[0], 15, 15), dtype=np.int8)
        for r, rr in vpairs:
            col_pair += x[:, r, :, None] == x[:, rr, None, :]
        block = np.empty((x.shape[0], len(hmaps)), dtype=np.int8)
        for hi, (hpairs, (dw, left)) in enumerate(hmaps):
            value = np.zeros(x.shape[0], dtype=np.int8)
            for c, cc in hpairs:
                value += col_pair[:, c, cc]
            block[:, hi] = value
            specs.append((dh, top, dw, left))
        blocks.append(block)
    return np.concatenate(blocks, axis=1), specs


def mask(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits).tobytes())


def find_disjoint(
    rels: list[tuple[int, str, int]], name: str, specs: list[tuple[int, int, int, int]],
    const: int,
) -> bool:
    by_mask: dict[int, tuple[int, str, int]] = {}
    for rel in rels:
        by_mask.setdefault(rel[2], rel)
    unique = sorted(by_mask.values(), key=lambda r: r[2].bit_count())
    for pos, a in enumerate(unique):
        for b in unique[pos + 1:]:
            if a[2] & b[2] == 0:
                print("PERFECT", name, "constant", const)
                print(a[1], specs[a[0]], const)
                print(b[1], specs[b[0]], const)
                return True
    return False


def main() -> None:
    x, req = examples(1_200, 343_172_517)
    only6 = req == 1
    only8 = req == -1
    fm, specs = build_features(x)
    print(
        "examples", len(req), "features", fm.shape[1],
        "only6", int(only6.sum()), "only8", int(only8.sum()), flush=True,
    )
    # These constants are themselves truthful scalar Conv correlations:
    # self-point=1, self-column=5, self-row=15, full self-grid=75.
    for const in (1, 5, 15, 75):
        universal: list[tuple[int, str, int]] = []
        zero8: list[tuple[int, str, int]] = []
        for op, pm in (("eq", fm == const), ("gt", fm > const), ("lt", fm < const)):
            good6 = np.all(pm[only6], axis=0)
            good8 = ~np.any(pm[only8], axis=0)
            both = np.flatnonzero(good6 & good8)
            if both.size:
                i = int(both[0])
                print("PERFECT SINGLE", op, specs[i], const)
                return
            for i in np.flatnonzero(good6):
                universal.append((int(i), op, mask(pm[only8, i])))
            for i in np.flatnonzero(good8):
                zero8.append((int(i), op, mask(~pm[only6, i])))
        print("constant", const, "universal", len(universal), "zero8", len(zero8), flush=True)
        if find_disjoint(universal, "AND", specs, const):
            return
        if find_disjoint(zero8, "OR", specs, const):
            return
    print("no exact sampled shared-constant formula")


if __name__ == "__main__":
    main()
