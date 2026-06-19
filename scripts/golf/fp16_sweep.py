#!/usr/bin/env python3
"""Sweep the general FP16 pass (g1_fp16_convert) across many tasks.

optimize_submission.py restricts G1 FP16 to a hardcoded 3 tasks (170/97/64),
but the pass itself is general and FP16 on big float intermediates is a proven
top-tier win (task169 +0.584). This sweeps it across a cost-filtered task set,
keeping only conversions that pass the SAME gates optimize_submission uses for
G1/G2:

  (a) masks_equal_with_margin(orig, fp16, 0.25)  -- outputs identical, margin>=0.25
  (b) cost(fp16) < cost(orig)                     -- strictly cheaper
  (c) score_and_verify(require_correct=True)      -- visible gold via scripts/lib

Survivors are written to a staging dir and printed with cost/score deltas. They
are NOT auto-banked: run verify_fix.py (fresh 5000 + official gold) on them
before merging — fresh-gate is mandatory (only-merge-fresh-verified).

Usage: fp16_sweep.py [--min-cost 12000] [--tasks 1,2,3] [--out artifacts/fp16_stage]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import dtype_passes as dtype_opt  # noqa: E402
from lib import scoring  # noqa: E402

BEST_ZIP = REPO / "artifacts" / "_BEST_6780.74.zip"
G2_MARGIN = 0.25
ALREADY_FP16 = {170, 97, 64}  # optimize_submission already handles these


def _score(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost))


def _cost(model: onnx.ModelProto, task: int, wd: str) -> int | None:
    s = scoring.score_and_verify(model, task, wd, label="c", require_correct=False)
    return s["cost"] if s else None


def try_fp16(task: int, base: onnx.ModelProto, wd: str) -> dict:
    orig_cost = _cost(base, task, wd)
    if orig_cost is None:
        return {"task": task, "ok": False, "reason": "base_unscorable"}
    try:
        fp16, _stats = dtype_opt.g1_fp16_convert(base)
    except Exception as e:  # noqa: BLE001
        return {"task": task, "ok": False, "reason": f"convert_err:{e!s:.40}"}
    # (a) output identity with margin
    try:
        if not scoring.masks_equal_with_margin(base, fp16, task, G2_MARGIN):
            return {"task": task, "ok": False, "reason": "margin/mask"}
    except Exception as e:  # noqa: BLE001
        return {"task": task, "ok": False, "reason": f"mask_err:{e!s:.30}"}
    # (c) visible gold (scripts/lib)
    scored = scoring.score_and_verify(fp16, task, wd, label="fp16", require_correct=True)
    if scored is None:
        return {"task": task, "ok": False, "reason": "not_gold"}
    new_cost = scored["cost"]
    # (b) strictly cheaper
    if new_cost >= orig_cost:
        return {"task": task, "ok": False, "reason": "not_cheaper",
                "orig_cost": orig_cost, "new_cost": new_cost}
    return {
        "task": task, "ok": True, "orig_cost": orig_cost, "new_cost": new_cost,
        "delta_score": _score(new_cost) - _score(orig_cost), "model": fp16,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cost", type=float, default=12000.0)
    ap.add_argument("--tasks", help="comma list; overrides --min-cost selection")
    ap.add_argument("--out", default="artifacts/fp16_stage")
    a = ap.parse_args()

    best = zipfile.ZipFile(BEST_ZIP)
    out_dir = REPO / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.tasks:
        tasks = [int(x) for x in a.tasks.split(",")]
    else:
        tasks = []
        with tempfile.TemporaryDirectory() as wd:
            for t in range(1, 401):
                n = f"task{t:03d}.onnx"
                if n not in best.namelist() or t in ALREADY_FP16:
                    continue
                try:
                    c = _cost(onnx.load_model_from_string(best.read(n)), t, wd)
                except Exception:  # noqa: BLE001
                    c = None
                if c and c >= a.min_cost:
                    tasks.append(t)
    print(f"sweeping FP16 over {len(tasks)} tasks (min_cost={a.min_cost})",
          file=sys.stderr)

    wins = []
    with tempfile.TemporaryDirectory() as wd:
        for t in tasks:
            base = onnx.load_model_from_string(best.read(f"task{t:03d}.onnx"))
            r = try_fp16(t, base, wd)
            if r["ok"]:
                onnx.save(r.pop("model"), str(out_dir / f"task{t:03d}.onnx"))
                wins.append(r)
                print(f"  WIN task{t:03d}: {r['orig_cost']} -> {r['new_cost']} "
                      f"({r['delta_score']:+.3f})", file=sys.stderr)
            else:
                print(f"  --  task{t:03d}: {r['reason']}", file=sys.stderr)

    wins.sort(key=lambda r: -r["delta_score"])
    total = sum(r["delta_score"] for r in wins)
    print(json.dumps({"wins": wins, "total_delta": total,
                      "staged_dir": str(out_dir)}, indent=2))
    print(f"\nFP16 WINS: {len(wins)} tasks, total +{total:.3f} "
          f"(staged in {a.out}; fresh-gate before banking)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
