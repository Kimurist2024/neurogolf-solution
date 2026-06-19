#!/usr/bin/env python3
"""Build an aggressive submission: sub12 baseline + cheapest visible-correct
local swap per task. NO fresh-gate (the public LB is the judge).

Per task:
  - baseline = the sub12 net (artifacts/sub12_base/taskXXX.onnx), always kept as
    the fallback (it is the official 6707.88 net; some tasks are locally
    divergent so we must NOT require local visible-gold on the baseline itself).
  - candidates = optimized/ and handcrafted/ nets. A candidate replaces the
    baseline ONLY if it is cheaper AND passes visible gold (train+test+arc-gen
    exact). This avoids guaranteed-zero regressions while skipping the fresh
    generalization gate on purpose.

Outputs:
  - artifacts/aggr_build/  (400 selected onnx)
  - artifacts/submission_aggr.zip
  - artifacts/aggr_manifest.json  (every changed task + projected delta)
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from lib import scoring  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402

SUB12 = REPO / "artifacts" / "sub12_base"
OPT = REPO / "artifacts" / "optimized"
HC = REPO / "artifacts" / "handcrafted"
BUILD = REPO / "artifacts" / "aggr_build"
ZIP = REPO / "artifacts" / "submission_aggr.zip"
MANIFEST = REPO / "artifacts" / "aggr_manifest.json"
SUB12_COSTS = REPO / "docs" / "golf" / "sub12_costs.json"
SIZE_LIMIT = 1_509_949  # 1.44 MiB


def sc(c: float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, c)))


def main() -> int:
    sub12_costs = json.load(open(SUB12_COSTS))["costs"]
    BUILD.mkdir(parents=True, exist_ok=True)
    for f in BUILD.glob("*.onnx"):
        f.unlink()

    changed = []
    total_delta = 0.0
    with tempfile.TemporaryDirectory() as wd:
        for t in range(1, 401):
            base = SUB12 / f"task{t:03d}.onnx"
            if not base.is_file():
                raise SystemExit(f"missing sub12 net for task{t:03d}")
            base_cost = int(sub12_costs[str(t)])
            best_path, best_cost, best_src = base, base_cost, "sub12"

            for cand, src in ((HC / f"task{t:03d}.onnx", "handcrafted"),
                              (OPT / f"task{t:03d}.onnx", "optimized")):
                if not cand.is_file():
                    continue
                # cheap static pre-filter: only bother verifying if plausibly cheaper
                stat_cost = cost_of(str(cand))[2]
                if stat_cost is None or stat_cost < 0 or stat_cost >= best_cost:
                    continue
                # visible-gold gate (require_correct=True); returns None if not exact
                res = scoring.score_and_verify(
                    onnx.load(str(cand)), t, wd, label="aggr", require_correct=True
                )
                if res and int(res["cost"]) < best_cost:
                    best_path, best_cost, best_src = cand, int(res["cost"]), src

            shutil.copy2(best_path, BUILD / f"task{t:03d}.onnx")
            if best_src != "sub12":
                d = sc(best_cost) - sc(base_cost)
                total_delta += d
                changed.append({
                    "task": t, "from": base_cost, "to": best_cost,
                    "src": best_src, "delta_score": round(d, 4),
                })

    files = sorted(BUILD.glob("task*.onnx"))
    assert len(files) == 400, f"expected 400, got {len(files)}"
    oversize = [f.name for f in files if f.stat().st_size > SIZE_LIMIT]
    assert not oversize, f"oversize files: {oversize}"

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)

    changed.sort(key=lambda c: -c["delta_score"])
    base_score = 6707.88
    payload = {
        "baseline": "submission (12).zip",
        "baseline_score": base_score,
        "n_changed": len(changed),
        "projected_delta": round(total_delta, 2),
        "projected_score": round(base_score + total_delta, 2),
        "changed": changed,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"built {ZIP.name}: {len(files)} nets, {len(changed)} swapped")
    print(f"projected: {base_score} + {total_delta:.2f} = {base_score + total_delta:.2f}")
    print("top swaps:")
    for c in changed[:20]:
        print(f"  task{c['task']:03d} {c['from']}->{c['to']} ({c['src']}) {c['delta_score']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
