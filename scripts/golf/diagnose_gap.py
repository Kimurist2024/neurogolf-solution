#!/usr/bin/env python3
"""Gap-diagnosis: after a bundle submission regresses, find the divergent task(s)
by computation (no bisection / no high-k / no source-name guessing).

A net that passes fresh-gate locally but is wrong on the private set scores 0 on
the grader, losing its full projected score. So:
    gap = (local projected score of bundle) - (actual LB)  ~=  divergent task's new_score
Rank changed tasks by |new_score - gap|; the closest is the culprit. Revert it
and resubmit (the FIX). For 2+ divergent tasks the gap is the sum (subset-sum);
revert the closest 1-2 and re-diagnose.

Usage: diagnose_gap.py --bundle <zip> --base <zip> --actual <LB_float> [--top 8]
       base defaults to the campaign_best pointer.
"""
from __future__ import annotations
import argparse, math, sys, tempfile, zipfile
from pathlib import Path
import onnx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402


def sc(c) -> float:
    return max(1.0, 25 - math.log(c)) if c and c > 0 else 0.0


def cost(z: zipfile.ZipFile, name: str, t: int, wd: str):
    s = scoring.score_and_verify(onnx.load_model_from_string(z.read(name)), t, wd,
                                 label="x", require_correct=False)
    return s["cost"] if s else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True, help="submitted bundle zip")
    ap.add_argument("--base", type=Path,
                    default=Path((REPO / "docs/golf/campaign_best.txt").read_text().split("\t")[0]))
    ap.add_argument("--base-lb", type=float, required=True, help="LB score of --base")
    ap.add_argument("--actual", type=float, required=True, help="actual LB of --bundle")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    base, bundle = zipfile.ZipFile(a.base), zipfile.ZipFile(a.bundle)
    rows, proj_gain = [], 0.0
    with tempfile.TemporaryDirectory() as wd:
        for n in sorted(x for x in bundle.namelist() if x.endswith(".onnx")):
            if bundle.read(n) == base.read(n):
                continue
            t = int(n[4:7])
            bc = cost(base, n, t, wd)
            nc = cost(bundle, n, t, wd)
            rows.append((t, bc, nc, sc(bc), sc(nc)))
            proj_gain += sc(nc) - sc(bc)

    projected = a.base_lb + proj_gain
    gap = projected - a.actual
    print(f"changed tasks: {len(rows)}")
    print(f"projected = base_lb {a.base_lb} + local_gain {proj_gain:+.3f} = {projected:.2f}")
    print(f"actual    = {a.actual:.2f}")
    print(f"GAP       = {gap:+.2f}  (~= sum of divergent tasks' new_score)\n")
    if gap < 0.3:
        print("gap < 0.3: no meaningful divergence; bundle is essentially as projected.")
        return 0
    rows.sort(key=lambda r: abs(r[4] - gap))
    print(f"prime suspects (|new_score - {gap:.2f}| ascending):")
    for t, bc, nc, bs, ns in rows[:a.top]:
        print(f"  task{t:>3}: new_cost={nc} new_score={ns:.3f}  |Δgap|={abs(ns-gap):.3f}  (base {bs:.3f})")
    print("\n-> revert the closest 1-2 to base, resubmit (FIX), re-diagnose if still low.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
