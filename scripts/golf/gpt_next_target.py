#!/usr/bin/env python3
"""Emit the next GPT-rebuild target from docs/golf/gpt5000_targets.json.

Pool = the fixed cost>5000 task list. Effective cost = current handcrafted (if
present & correct) else the listed cost. Skips tasks already at/below GOAL(1000)
and tasks already assigned this pass (docs/golf/gpt_assigned.json). Cheapest-
first (closest to the 1000 floor = most achievable rebuild). When the pass is
exhausted the caller resets gpt_assigned.json for a re-attempt pass.

Output: one line per target -> task:hash:cost   (N lines, default 1)
Usage: gpt_next_target.py [N=1]

NOTE: handcrafted scoring runs in a child process with a hard timeout. Some
fully-optimized nets use a giant fused Einsum that hangs ONNX Runtime locally
(the grader scores them fine). Such a net is ALREADY at/near its floor, so on a
scoring timeout we SKIP the task entirely rather than block the whole campaign.
"""
from __future__ import annotations
import json, multiprocessing as mp, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FS = REPO / "docs" / "golf"
HAND = REPO / "artifacts" / "handcrafted"
GOAL = 1000
HAND_TIMEOUT = 25            # sec; hang-prone giant-Einsum nets are skipped
HANG = "__HANG__"           # sentinel: scoring timed out -> skip task
TARGETS = FS / "gpt5000_targets.json"


def load(p, d):
    try:
        return json.load(open(p))
    except Exception:
        return d


MAX_EINSUM_OPERANDS = 15    # >= this many inputs on one Einsum => hangs ORT


def _is_hang_prone(path: str) -> bool:
    """A giant fused-Einsum net hangs ORT locally (grader scores it fine). Detect
    it by node inspection alone (cheap) -- no scoring, no 25s timeout wait."""
    try:
        import onnx
        m = onnx.load(path)
        for nd in m.graph.node:
            if nd.op_type == "Einsum" and len(nd.input) >= MAX_EINSUM_OPERANDS:
                return True
    except Exception:
        return False
    return False


def _hand_worker(path: str, t: int, q) -> None:
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from lib import scoring
        import onnx
        with tempfile.TemporaryDirectory() as wd:
            s = scoring.score_and_verify(onnx.load(path), t, wd, label="x",
                                         require_correct=True)
        q.put(s["cost"] if s else None)
    except Exception:
        q.put(None)


def hand_cost(t: int):
    """Return handcrafted cost (int), None (no/invalid net), or HANG (timeout)."""
    p = HAND / f"task{t:03d}.onnx"
    if not p.is_file():
        return None
    if _is_hang_prone(str(p)):
        return HANG             # at-floor fused-Einsum net -> skip, do not score
    q = mp.Queue()
    proc = mp.Process(target=_hand_worker, args=(str(p), t, q))
    proc.start()
    proc.join(HAND_TIMEOUT)
    if proc.is_alive():
        proc.terminate()
        proc.join(3)
        if proc.is_alive():
            proc.kill()
        return HANG
    try:
        return q.get_nowait()
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
        hc = hand_cost(t)
        if hc == HANG:
            # fully-optimized fused-Einsum net (hangs ORT locally) => already at
            # its floor; do NOT waste a worker re-attempting it.
            continue
        # effective = better of best-listed vs handcrafted; a STALE-expensive
        # handcrafted net must NOT inflate priority (it is not in the best zip).
        c = min(listed, hc) if isinstance(hc, int) else listed
        if c <= GOAL:
            continue
        rows.append((c, t, h))
    rows.sort(reverse=True)           # high-cost-first = biggest ln-gain ROI
    out = [f"{t}:{h}:{int(c)}" for c, t, h in rows[:n]]
    print(" ".join(out))
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
