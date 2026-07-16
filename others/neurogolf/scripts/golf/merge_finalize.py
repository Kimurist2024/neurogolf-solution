#!/usr/bin/env python3
"""Fresh-gate the multi_merge_scan staged winners and build the merged submission.

Pipeline (safe-merge per only-merge-fresh-verified):
  1. multi_merge_scan.py has staged per-task cheapest CORRECT challengers
     (cheaper than the base) to artifacts/merge_stage/task<NNN>.onnx + manifest.
  2. Here we FRESH-GATE each staged winner with verify_fix.verify_one (k default
     30): lib gold + official neurogolf_utils gold + margin stability + a
     k-instance fresh generator audit with ZERO failures. Only ADOPT is kept.
  3. Build <base> + adopted swaps -> artifacts/merge_finalized.zip.
  4. Full-score base vs merged (require_correct=False, both grader-style) and
     print the per-task adopted deltas and projected total.

Does NOT submit. Review the projection, then submit separately.

Usage: merge_finalize.py [--k 30] [--base artifacts/_BEST_6814.95.zip]
"""
from __future__ import annotations
import argparse, json, math, sys, tempfile, zipfile
from pathlib import Path
import onnx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import verify_fix  # noqa: E402
from lib import scoring  # noqa: E402

STAGE = REPO / "artifacts" / "merge_stage"


def score(c) -> float:
    return max(1.0, 25 - math.log(c)) if c and c > 0 else 1.0


def full_score(zip_path: Path):
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if n.endswith(".onnx")]
    total, costs = 0.0, {}
    with tempfile.TemporaryDirectory() as wd:
        for n in sorted(names):
            t = int(n[4:7])
            m = onnx.load_model_from_string(z.read(n))
            s = scoring.score_and_verify(m, t, wd, label="x", require_correct=False)
            if s:
                costs[t] = s["cost"]
                total += score(s["cost"])
    return total, costs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--base", type=Path, default=REPO / "artifacts" / "_BEST_6814.95.zip")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "merge_finalized.zip")
    ap.add_argument("--max-base-cost", type=int, default=0,
                    help="exclude winners whose BASE cost exceeds this (monster-cut "
                         "private-divergence guard; 0 = no limit)")
    a = ap.parse_args()

    manifest = json.loads((STAGE / "manifest.json").read_text())
    print(f"staged winners: {len(manifest)}  (fresh-gate k={a.k})\n")

    adopts, rejects, monster = [], [], []
    for w in sorted(manifest, key=lambda r: -r["delta"]):
        t = w["task"]
        hp = STAGE / f"task{t:03d}.onnx"
        if not hp.is_file():
            continue
        if a.max_base_cost and w["best"] > a.max_base_cost:
            monster.append((t, w))
            print(f"  MONSTER-SKIP task{t:>3}: base={w['best']} > {a.max_base_cost} "
                  f"[{w['source']}] (private-divergence guard; not fresh-gated)")
            continue
        v = verify_fix.verify_one(t, hp, a.k)
        if v["decision"] == "ADOPT":
            adopts.append((t, w, v))
            print(f"  ADOPT  task{t:>3}: {w['best']:>7} -> {v['cost']:>7} "
                  f"[{w['source']:>9}] +{score(v['cost'])-score(w['best']):.3f} "
                  f"fresh={v['fresh_fails']}/{v['fresh_total']}")
        else:
            rejects.append((t, w, v))
            why = (f"fresh={v.get('fresh_fails')}/{v.get('fresh_total')} "
                   f"lib={v.get('lib_gold')} off={v.get('official_gold')} "
                   f"stable={v.get('margin_stable')}")
            print(f"  REJECT task{t:>3}: [{w['source']:>9}] {why}")

    src = zipfile.ZipFile(a.base)
    names = sorted(n for n in src.namelist() if n.endswith(".onnx"))
    swaps = {t: (STAGE / f"task{t:03d}.onnx").read_bytes() for t, _, _ in adopts}
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            t = int(n[4:7])
            z.writestr(n, swaps.get(t, src.read(n)))

    base_total, _ = full_score(a.base)
    new_total, _ = full_score(a.out)
    proj = sum(score(v["cost"]) - score(w["best"]) for _, w, v in adopts)
    by_src = {}
    for _, w, _v in adopts:
        by_src[w["source"]] = by_src.get(w["source"], 0) + 1
    print(f"\nADOPT {len(adopts)}  REJECT {len(rejects)}  MONSTER-SKIP {len(monster)}"
          f"  by-source {by_src}")
    print(f"base   full-score {base_total:.2f}")
    print(f"merged full-score {new_total:.2f}   (Δ {new_total - base_total:+.2f}; "
          f"adopted-delta sum +{proj:.2f})")
    print(f"out: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
