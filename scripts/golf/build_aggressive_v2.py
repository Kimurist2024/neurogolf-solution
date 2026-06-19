#!/usr/bin/env python3
"""Aggressive submission builder v2 — REAL-input costs only (no zero-input bug).

Reads the official-faithful full-score JSONs (docs/golf/fullscore_*.json) produced
by score_dir_full.py for sub12_base / optimized / handcrafted. Per task:
  - baseline = sub12 net, base_cost = its REAL cost (always kept as fallback;
    sub12 is officially correct even where locally divergent).
  - a candidate (handcrafted/optimized) replaces it only if it is LOCALLY
    visible-correct AND its REAL cost < base_cost.
This eliminates the zero-input mis-measurement that made v1 wrongly swap task188
(real sub12 cost 3440) for an 8103 net.
"""
from __future__ import annotations
import json, math, shutil, sys, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUB12 = REPO / "artifacts" / "sub12_base"
OPT = REPO / "artifacts" / "optimized"
HC = REPO / "artifacts" / "handcrafted"
BUILD = REPO / "artifacts" / "aggr_build_v2"
ZIP = REPO / "artifacts" / "submission_aggr_v2.zip"
MANIFEST = REPO / "artifacts" / "aggr_manifest_v2.json"
FS = REPO / "docs" / "golf"
SIZE_LIMIT = 1_509_949


def sc(c): return max(1.0, 25.0 - math.log(max(1.0, c)))


def main() -> int:
    sub = json.load(open(FS / "fullscore_sub12_base.json"))
    opt = json.load(open(FS / "fullscore_optimized.json"))
    hand = json.load(open(FS / "fullscore_handcrafted.json"))
    BUILD.mkdir(parents=True, exist_ok=True)
    for f in BUILD.glob("*.onnx"):
        f.unlink()

    changed, base_total, new_total = [], 0.0, 0.0
    for t in range(1, 401):
        ts = str(t)
        base_cost = sub.get(ts, {}).get("cost")
        if base_cost is None:
            raise SystemExit(f"missing real sub12 cost for task{t:03d}")
        best_path, best_cost, best_src = SUB12 / f"task{t:03d}.onnx", base_cost, "sub12"

        for d, src in ((hand, "handcrafted"), (opt, "optimized")):
            r = d.get(ts)
            if not r or r.get("cost") is None or not r.get("correct"):
                continue  # must be locally visible-correct to be swap-eligible
            if r["cost"] < best_cost:
                srcdir = HC if src == "handcrafted" else OPT
                best_path, best_cost, best_src = srcdir / f"task{t:03d}.onnx", int(r["cost"]), src

        shutil.copy2(best_path, BUILD / f"task{t:03d}.onnx")
        base_total += sc(base_cost)
        new_total += sc(best_cost)
        if best_src != "sub12":
            changed.append({"task": t, "from": base_cost, "to": best_cost,
                            "src": best_src, "delta_score": round(sc(best_cost) - sc(base_cost), 4)})

    files = sorted(BUILD.glob("task*.onnx"))
    assert len(files) == 400, f"expected 400, got {len(files)}"
    assert not [f for f in files if f.stat().st_size > SIZE_LIMIT]
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)

    delta = new_total - base_total
    changed.sort(key=lambda c: -c["delta_score"])
    payload = {
        "baseline_score": 6707.88,
        "method": "real-input costs, no zero-input",
        "n_changed": len(changed),
        "real_projected_delta": round(delta, 2),
        "real_projected_score": round(6707.88 + delta, 2),
        "changed": changed,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"built {ZIP.name}: {len(files)} nets, {len(changed)} swapped")
    print(f"REAL projected: 6707.88 + {delta:.2f} = {6707.88 + delta:.2f}")
    print("top swaps:")
    for c in changed[:25]:
        print(f"  task{c['task']:03d} {c['from']}->{c['to']} ({c['src']}) {c['delta_score']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
