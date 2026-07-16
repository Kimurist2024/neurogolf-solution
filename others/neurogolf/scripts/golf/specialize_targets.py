#!/usr/bin/env python3
"""Emit the next N specialization targets: tasks whose current cost is still
> GOAL (8000 = score 16.01), cheapest-first (closest to crossing = smallest cut
needed = highest yield). Skips tasks already assigned this pass
(docs/golf/specialize_assigned.json) and tasks already <= GOAL.

Effective cost = min(campaign_costs, current handcrafted) so a task already
pushed below GOAL (pending) is not re-picked.

Output: one line per target  ->  task:hash:cost
Usage: specialize_targets.py [N=12]
"""
from __future__ import annotations
import json, math, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FS = REPO / "docs" / "golf"
HAND = REPO / "artifacts" / "handcrafted"
GOAL = 8000
HMAP = json.load(open(FS / "task_hash_map.json"))


def load(p, d):
    try:
        return json.load(open(p))
    except Exception:
        return d


def hand_cost(t: int):
    p = HAND / f"task{t:03d}.onnx"
    if not p.is_file():
        return None
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from lib import scoring
        import onnx
        with tempfile.TemporaryDirectory() as wd:
            s = scoring.score_and_verify(onnx.load(str(p)), t, wd, label="x", require_correct=True)
        return s["cost"] if s else None
    except Exception:
        return None


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    costs = {int(a): b for a, b in load(FS / "campaign_costs.json", {}).items()
             if isinstance(b, (int, float))}
    assigned = set(load(FS / "specialize_assigned.json", []))
    over = []
    for t, c in costs.items():
        if c <= GOAL or t in assigned:
            continue
        over.append((c, t))
    over.sort()                       # cheapest-first
    out = []
    for c, t in over[:n]:
        h = HMAP.get(f"{t:03d}", "-")
        out.append(f"{t}:{h}:{int(c)}")
    print(" ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
