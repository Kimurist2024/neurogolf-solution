#!/usr/bin/env python3
"""Pick N Kimi golf targets: the task12-type cheap/simple band, cheapest first.

Strategy (per user): score = max(1, 25-ln(cost)), so d(score)/d(cost) = -1/cost.
At LOW cost a small absolute cut earns a full +1 point, so the cheap simple
tasks (task12 archetype, cost ~1220) are where +1/task is reliably achievable.
We march the cheap band ASCENDING so each +1 needs the smallest absolute cut,
toward the "+1 on all 400 => +400" goal.

OPEN = not in docs/golf/kimi_exclude.json (factory_done + recent codex waves,
lane separation) and not in docs/golf/kimi_attempted.json. Uses the real-input
cost ranking (docs/golf/real_incumbent.json), NOT the buggy zero-input one.
Tasks already below --min-cost can't gain +1 (near floor) so are skipped.
Prints space-separated `task:hash:cost` triples.

Usage: kimi_targets.py [N=3] [--max-cost C(default 25000)] [--min-cost C(default 850)]
"""
from __future__ import annotations
import json, sys
from pathlib import Path

FS = Path(__file__).resolve().parents[2] / "docs" / "golf"


def main() -> int:
    n = 3
    max_cost = 25000.0   # band ceiling (solo owns >=19931 via kimi_exclude, so the
                         # effective kimi ceiling is the solo boundary ~19930)
    min_cost = 4000.0    # SWITCHED UP (2026-06-15): the ~900-4000 near-floor cluster
                         # only yielded PARTIAL (<1.0) gains; move to the compressible
                         # mid band (4000-19930) where +1 is achievable
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--max-cost":
            max_cost = float(args[i + 1]); i += 2
        elif args[i] == "--min-cost":
            min_cost = float(args[i + 1]); i += 2
        else:
            n = int(args[i]); i += 1

    inc = json.load(open(FS / "real_incumbent.json"))
    hashes = json.load(open(FS / "task_hash_map.json"))
    exclude = set(json.load(open(FS / "kimi_exclude.json")))
    attempted = set(json.load(open(FS / "kimi_attempted.json")))
    blocked = exclude | attempted

    # cheapest-first within the band: smallest absolute cut earns the next +1
    ranked = sorted(((int(t), c) for t, c in inc.items()), key=lambda kv: kv[1])
    out = []
    for t, c in ranked:
        if t in blocked:
            continue
        if not (min_cost <= c <= max_cost):
            continue
        h = hashes.get(f"{t:03d}") or hashes.get(str(t))
        if not h:
            continue
        out.append(f"{t}:{h}:{int(c)}")
        if len(out) >= n:
            break
    print(" ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
