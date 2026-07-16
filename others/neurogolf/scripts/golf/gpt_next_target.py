#!/usr/bin/env python3
"""Emit the next GPT-rebuild target from docs/golf/gpt5000_targets.json.

Pool = the fixed cost>5000 task list. Effective cost = current handcrafted (if
present & correct) else the listed cost. Skips tasks already at/below GOAL(1000)
and tasks already assigned this pass (docs/golf/gpt_assigned.json). Cheapest-
first (closest to the 1000 floor = most achievable rebuild). When the pass is
exhausted the caller resets gpt_assigned.json for a re-attempt pass.

Output: one line per target -> task:hash:cost   (N lines, default 1)
Usage: gpt_next_target.py [N=1]
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FS = REPO / "docs" / "golf"
HAND = REPO / "artifacts" / "handcrafted"
GOAL = 1000
TARGETS = FS / "gpt5000_targets.json"


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
            s = scoring.score_and_verify(onnx.load(str(p)), t, wd, label="x",
                                         require_correct=True)
        return s["cost"] if s else None
    except Exception:
        return None


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    pool = load(TARGETS, [])
    assigned = set(load(FS / "gpt_assigned.json", []))
    rows = []
    for g in pool:
        t, h, listed = int(g["task"]), g["hash"], int(g["cost"])
        if t in assigned:
            continue
        c = hand_cost(t)
        c = c if c is not None else listed
        if c <= GOAL:
            continue
        rows.append((c, t, h))
    rows.sort()                       # cheapest-first = closest to GOAL
    out = [f"{t}:{h}:{int(c)}" for c, t, h in rows[:n]]
    print(" ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
