"""Scan multiple submission sources, pick cheapest CORRECT version per task,
stage winners (cheaper than the grader-proven best) for fresh-gating.

Sources (challengers must be locally gold-correct; best is the baseline):
  best     = artifacts/_BEST_6780.74.zip   (baseline, require_correct=False)
  zip1/2/3 = the user-supplied submission zips
  hand     = artifacts/handcrafted/<task>.onnx  (per-task override)

Winners (cheaper+correct) are written to artifacts/merge_stage/task<NNN>.onnx with
a sidecar manifest. NOT adopted until verify_fix.py fresh-gates them.
"""
from __future__ import annotations
import json, math, sys, tempfile, zipfile
from pathlib import Path
import onnx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402

BEST = REPO / "artifacts" / "_BEST_6830.zip"
HAND = REPO / "artifacts" / "handcrafted"
STAGE = REPO / "artifacts" / "merge_stage"
OTH = REPO / "others"
ZIPS = {
    "s9": OTH / "submission (9).zip",
    "s10": OTH / "submission (10).zip",
    "no179_p6": OTH / "submission6824.62_no179241_newtasks_plus6.178.zip",
    "no179_p055": OTH / "submission6824.62_no179241_plus0p55.zip",
    "p45_v5": OTH / "submission6824.62_plus45_fresh90_v5.zip",
    "relax90": OTH / "submission6824.62_relaxed90_more_plus_iter.zip",
    "anchor179": OTH / "submission6824_62_newtasks_anchor179_241_plus50.zip",
    "p14_074": OTH / "submission6824_62_plus14_074_newtasks.zip",
    "struct": OTH / "submission6824_structural_improved.zip",
    "b6834": OTH / "submission6834_722_fresh90_plus10_attempt.zip",
    "bounce": OTH / "submission_bounce_plus_248_357.zip",
}


def _cost(model, t, req, wd):
    s = scoring.score_and_verify(model, t, wd, label="x", require_correct=req)
    return s["cost"] if s else None


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    best = zipfile.ZipFile(BEST)
    zips = {k: zipfile.ZipFile(p) for k, p in ZIPS.items() if p.is_file()}
    wins = []
    with tempfile.TemporaryDirectory() as wd:
        for t in range(1, 401):
            n = f"task{t:03d}.onnx"
            bc = _cost(onnx.load_model_from_string(best.read(n)), t, False, wd)
            if bc is None:
                continue
            cands = []  # (cost, source, model)
            for k, zf in zips.items():
                if n in zf.namelist():
                    m = onnx.load_model_from_string(zf.read(n))
                    c = _cost(m, t, True, wd)
                    if c:
                        cands.append((c, k, m))
            hp = HAND / n
            if hp.is_file():
                m = onnx.load(str(hp))
                c = _cost(m, t, True, wd)
                if c:
                    cands.append((c, "hand", m))
            if not cands:
                continue
            cands.sort(key=lambda r: r[0])
            wc, ws, wm = cands[0]
            if wc < bc - 1:
                onnx.save(wm, str(STAGE / n))
                d = max(1, 25 - math.log(wc)) - max(1, 25 - math.log(bc))
                wins.append({"task": t, "best": bc, "win_cost": wc,
                             "source": ws, "delta": round(d, 4)})
    wins.sort(key=lambda r: -r["delta"])
    total = sum(w["delta"] for w in wins)
    (STAGE / "manifest.json").write_text(json.dumps(wins, indent=2))
    for w in wins:
        print(f"  task{w['task']:>3}: {w['best']:>7} -> {w['win_cost']:>7} "
              f"[{w['source']}] {w['delta']:+.3f}")
    print(f"\nWINNERS: {len(wins)} tasks, total +{total:.3f} "
          f"(staged in {STAGE}; FRESH-GATE before adopt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
