#!/usr/bin/env python3
"""Test cheap row-score formulas against task396 generator geometry."""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))


def rects(a: np.ndarray, color: int):
    h, w = a.shape
    seen = set()
    result = []
    for rr, cc in zip(*np.where(a == color)):
        start = (int(rr), int(cc))
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = set()
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and a[nr, nc] == color:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if comp == border and r1 - r0 >= 2 and c1 - c0 >= 2:
            result.append((r0, r1, c0, c1))
    return result


def lscore(a: np.ndarray, color: int) -> int:
    mask = a == color
    rows = mask.sum(axis=1)
    cols = mask.sum(axis=0)
    return int(np.sum(mask * rows[:, None] * cols[None, :]))


def scores(counts: np.ndarray, kind: str) -> np.ndarray:
    z = np.pad(counts.astype(np.float64), (1, 1))
    prev, cur, nxt = z[:-2], z[1:-1], z[2:]
    if kind == "laplace":
        return 2 * cur - prev - nxt
    if kind == "absdiff":
        return np.abs(cur - prev) + np.abs(cur - nxt)
    if kind == "peak":
        return cur - np.minimum(prev, nxt)
    if kind == "minedge":
        return np.minimum(cur - prev, cur - nxt)
    if kind == "count":
        return cur
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=404_039_600)
    args = parser.parse_args()
    gen = importlib.import_module("task_fcb5c309")
    kinds = ("laplace", "absdiff", "peak", "minedge", "count")
    stats = defaultdict(int)
    examples = {}
    random.seed(args.seed)
    for index in range(args.count):
        ex = gen.generate()
        a = np.asarray(ex["input"], dtype=np.int64)
        colors = [int(v) for v in np.unique(a) if v]
        box_color = max(colors, key=lambda c: lscore(a, c))
        found = rects(a, box_color)
        if not found:
            stats["rect_parse_fail"] += 1
            continue
        true = max(found, key=lambda x: (x[3] - x[2] + 1) * (x[1] - x[0] + 1))
        r0, r1, _, _ = true
        row_counts = (a == box_color).sum(axis=1)
        for kind in kinds:
            sc = scores(row_counts, kind)
            order = np.argsort(-sc, kind="stable")
            for k in (2, 3, 4):
                ok = r0 in order[:k] and r1 in order[:k]
                key = f"{kind}_k{k}_{'ok' if ok else 'fail'}"
                stats[key] += 1
                if not ok and key not in examples:
                    examples[key] = {
                        "index": index,
                        "shape": list(a.shape),
                        "box_color": box_color,
                        "true_rows": [r0, r1],
                        "counts": row_counts.tolist(),
                        "scores": sc.tolist(),
                        "top": order[:k].tolist(),
                    }
    import json

    result = {"count": args.count, "seed": args.seed, "stats": dict(stats), "first_failures": examples}
    (HERE / "row_score_study.json").write_text(json.dumps(result, indent=2) + "\n")
    for kind in kinds:
        print(kind, {k: stats[f"{kind}_k{k}_fail"] for k in (2, 3, 4)})


if __name__ == "__main__":
    main()
