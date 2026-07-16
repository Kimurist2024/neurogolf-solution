#!/usr/bin/env python3
"""Score a whole submission dir the official way (real-input profiling) and sum.

For each taskXXX.onnx: score_and_verify(require_correct=False) -> real cost +
local visible-correctness. Reports:
  - sum of 25-ln(cost) over ALL 400 (optimistic: assumes every net is correct)
  - the subset that FAILS local visible gold (these would score differently)
This is the calibration check: local optimistic sum should match the official
public_score when the submission has no overfit and cost matches Kaggle.
"""
from __future__ import annotations
import json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import onnx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402

def sc(c): return max(1.0, 25.0 - math.log(max(1.0, c)))

def _job(item):
    t, path = item
    import tempfile
    with tempfile.TemporaryDirectory() as wd:
        try:
            r = scoring.score_and_verify(onnx.load(path), t, wd, label="full", require_correct=False)
        except Exception as e:
            return (t, None, None, f"ERR {e}")
    if r is None:
        return (t, None, None, "UNSCORABLE")
    return (t, int(r["cost"]), bool(r["correct"]), "")

def main():
    d = Path(sys.argv[1])
    items = []
    for t in range(1, 401):
        p = d / f"task{t:03d}.onnx"
        if p.is_file():
            items.append((t, str(p)))
    rows = {}
    workers = max(1, (os.cpu_count() or 4) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for t, cost, correct, note in ex.map(_job, items):
            rows[t] = {"cost": cost, "correct": correct, "note": note}
    total_optimistic = 0.0
    total_localcorrect = 0.0
    incorrect = []
    unscorable = []
    for t in range(1, 401):
        r = rows.get(t)
        if not r or r["cost"] is None:
            unscorable.append((t, r["note"] if r else "MISSING"))
            continue
        s = sc(r["cost"])
        total_optimistic += s
        if r["correct"]:
            total_localcorrect += s
        else:
            incorrect.append((t, r["cost"], round(s, 3)))
    print(f"dir={d.name}  nets={len(items)}")
    print(f"  SUM 25-ln(cost) over all (optimistic, assumes all correct): {total_optimistic:.2f}")
    print(f"  SUM over locally-visible-CORRECT only:                      {total_localcorrect:.2f}")
    print(f"  locally-INCORRECT (visible gold fail) count: {len(incorrect)}")
    for t, c, s in sorted(incorrect, key=lambda x: -x[2])[:20]:
        print(f"     task{t:03d} cost={c} score_if_correct={s}")
    if unscorable:
        print(f"  UNSCORABLE/missing: {unscorable[:20]}")
    out = REPO / "docs" / "golf" / f"fullscore_{d.name}.json"
    out.write_text(json.dumps({str(t): rows[t] for t in rows}, indent=2) + "\n")
    print(f"  wrote {out}")

if __name__ == "__main__":
    main()
