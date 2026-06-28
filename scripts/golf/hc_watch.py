#!/usr/bin/env python3
"""One pass: report handcrafted nets that now beat the campaign-best snapshot.

Reads /tmp/hc_best.json (task->best_cost reference) and /tmp/hc_emitted.json
(task->last_emitted_cost, to avoid re-emitting). For each target, scores
artifacts/handcrafted/taskNNN.onnx (per-model alarm timeout); if it is correct
and strictly cheaper than both the best snapshot and the last emitted cost,
prints one PROMOTED line and records it. Meant to be called in a poll loop by a
Monitor. The MAIN session harvests emitted tasks (fresh-gate -> merge -> submit)
and then bumps /tmp/hc_best.json for that task so it stops re-emitting.
"""
from __future__ import annotations
import sys, os, json, signal, tempfile, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402
import onnx  # noqa: E402

HAND = REPO / "artifacts" / "handcrafted"
BEST = Path("/tmp/hc_best.json")
EMIT = Path("/tmp/hc_emitted.json")


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))


def score(p: Path, t: int):
    signal.alarm(40)
    try:
        r = scoring.score_and_verify(onnx.load(str(p)), t, tempfile.mkdtemp(),
                                     label="x", require_correct=False)
        signal.alarm(0)
        return r
    except Exception:
        signal.alarm(0)
        return None


def main() -> int:
    best = json.load(open(BEST)) if BEST.exists() else {}
    emitted = json.load(open(EMIT)) if EMIT.exists() else {}
    changed = False
    for ts, bc in best.items():
        t = int(ts)
        p = HAND / f"task{t:03d}.onnx"
        if not p.is_file():
            continue
        r = score(p, t)
        if not r or not r.get("correct") or r.get("cost") is None:
            continue
        c = int(r["cost"])
        prev = emitted.get(ts)
        if c < int(bc) and (prev is None or c < int(prev)):
            print(f"PROMOTED task{t:03d} {int(bc)}->{c} (gain~{(math.log(max(1,int(bc)))-math.log(max(1,c))):.3f})", flush=True)
            emitted[ts] = c
            changed = True
    if changed:
        json.dump(emitted, open(EMIT, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
