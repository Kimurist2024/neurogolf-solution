#!/usr/bin/env python3
"""Cluster all 400 tasks by generator archetype to find craft-once-apply-many
opportunities (the "next rowcol" — one cheap ONNX pattern serving a homogeneous
high-cost cluster).

Archetype signal = the set of distinctive ``common.<helper>`` calls in each
task's generator spec (inputs/arc-gen-repo/tasks/task_<hash>.py), plus a few
example-derived structural flags (shape relation, recolor-only, symmetry). Ubiquitous
helpers (grids/randint/random_color/...) are dropped so the signature reflects
the TRANSFORM, not boilerplate. Tasks are grouped by Jaccard>=THRESHOLD on their
distinctive-helper sets (connected components). Clusters are ranked by aggregate
headroom = sum over members of (score_to_floor) so the highest-value homogeneous
clusters surface first.

Cost source: docs/golf/real_incumbent.json (approximate/stale — used only for
ranking, not correctness). Prints clusters of size>=MIN with members+costs.

Usage: archetype_classify.py [--min-size 3] [--jaccard 0.6] [--top 20]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "inputs" / "arc-gen-repo" / "tasks"
FS = REPO / "docs" / "golf"

# Helpers so common they carry no archetype signal.
UBIQUITOUS = {
    "grids", "grid", "randint", "randints", "random_color", "random_colors",
    "choice", "choices", "sample", "shuffle", "random_el", "deepcopy", "flatten",
    "black", "blue", "red", "green", "yellow", "gray", "pink", "orange", "cyan",
    "maroon", "set_colors", "remove_duplicates", "isclose", "sqrt", "int_sqrt",
}
CALL_RE = re.compile(r"common\.([a-z_]+)\s*\(")


def helper_set(task_hash: str) -> frozenset[str]:
    p = TASKS / f"task_{task_hash}.py"
    if not p.is_file():
        return frozenset()
    src = p.read_text(encoding="utf-8", errors="ignore")
    calls = {m.group(1) for m in CALL_RE.finditer(src)}
    return frozenset(calls - UBIQUITOUS)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    hashes = json.load(open(FS / "task_hash_map.json"))
    inc = json.load(open(FS / "real_incumbent.json"))

    sig: dict[int, frozenset[str]] = {}
    for t in range(1, 401):
        h = hashes.get(f"{t:03d}") or hashes.get(str(t))
        if h:
            s = helper_set(h)
            if s:
                sig[t] = s

    tasks = list(sig)
    # Union-find clustering on Jaccard>=threshold edges.
    parent = {t: t for t in tasks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, t in enumerate(tasks):
        for u in tasks[i + 1:]:
            if jaccard(sig[t], sig[u]) >= a.jaccard:
                union(t, u)

    clusters: dict[int, list[int]] = {}
    for t in tasks:
        clusters.setdefault(find(t), []).append(t)

    def cost(t):
        return float(inc.get(str(t), 0))

    def floor_head(t):
        c = cost(t)
        return max(0.0, (25 - math.log(400)) - max(1, 25 - math.log(c))) if c > 0 else 0.0

    rows = []
    for members in clusters.values():
        if len(members) < a.min_size:
            continue
        members = sorted(members, key=lambda t: -cost(t))
        head = sum(floor_head(t) for t in members)
        hi = [t for t in members if cost(t) >= 10000]
        # signature = most common distinctive helpers across the cluster
        common_helpers = Counter()
        for t in members:
            common_helpers.update(sig[t])
        tag = ",".join(h for h, _ in common_helpers.most_common(4))
        rows.append((head, len(members), len(hi), members, tag))

    rows.sort(key=lambda r: -r[0])
    print(f"{'headroom':>8} {'size':>4} {'#hi':>3}  signature / members")
    for head, n, nhi, members, tag in rows[:a.top]:
        ms = " ".join(f"{t}({int(cost(t)/1000)}k)" for t in members[:14])
        print(f"{head:>8.1f} {n:>4} {nhi:>3}  [{tag}]")
        print(f"             {ms}")
    print(f"\nclusters>= {a.min_size}: {len(rows)} | total tasks signed: {len(tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
