"""Global best-of merge across the 6484.74 base + sub29 + sub30 + wave2 handcrafted.

For every task that any source changes vs the base, pick the CHEAPEST candidate
that is locally gold-correct AND margin-stable. The base (grader-confirmed
6484.74) is always eligible via require_correct=False. Emits the winner set and
writes the merged ONNX files into an output dir.
"""
from __future__ import annotations
import sys, json, tempfile, shutil, filecmp
from pathlib import Path
import onnx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402

BASE = REPO / "artifacts" / "wave_opus" / "stage_backup_pre_wave2merge"  # 6484.74
OUT = REPO / "artifacts" / "bestof_out"
SOURCES = {
    "sub29": REPO / "artifacts" / "src_sub29",
    "sub30": REPO / "artifacts" / "src_sub30",
    "wave2": REPO / "artifacts" / "handcrafted",
}
# wave2 handcrafted dir holds ALL handcrafted nets; only treat these 7 as wave2 challengers
WAVE2_TASKS = {280, 379, 243, 23, 71, 77, 90}


def cost_of(path: Path, task: int, wd: str, require_correct: bool):
    if not path.exists():
        return None
    r = scoring.score_and_verify(onnx.load(str(path)), task, wd, label="c", require_correct=require_correct)
    if not r:
        return None
    if require_correct and not r["correct"]:
        return None
    return r


def main() -> int:
    # union of changed tasks across sources
    changed = set()
    base_files = {int(p.stem[4:]): p for p in BASE.glob("task*.onnx")}
    for name, d in SOURCES.items():
        for t, bp in base_files.items():
            if name == "wave2" and t not in WAVE2_TASKS:
                continue
            sp = d / f"task{t:03d}.onnx"
            if sp.exists() and not filecmp.cmp(sp, bp, shallow=False):
                changed.add(t)
    changed = sorted(changed)

    winners = []
    OUT.mkdir(parents=True, exist_ok=True)
    # start OUT as a full copy of base
    for bp in base_files.values():
        shutil.copy2(bp, OUT / bp.name)

    with tempfile.TemporaryDirectory() as wd:
        for t in changed:
            bp = base_files[t]
            base_r = cost_of(bp, t, wd, require_correct=False)
            best = ("base", base_r["cost"] if base_r else None, bp)
            margins = {}
            for name, d in SOURCES.items():
                if name == "wave2" and t not in WAVE2_TASKS:
                    continue
                sp = d / f"task{t:03d}.onnx"
                if not sp.exists() or filecmp.cmp(sp, bp, shallow=False):
                    continue
                r = cost_of(sp, t, wd, require_correct=True)
                if not r:
                    continue
                stable, mm = scoring.model_margin_stable(onnx.load(str(sp)), t)
                margins[name] = (r["cost"], r["correct"], stable, mm)
                if stable and best[1] is not None and r["cost"] < best[1]:
                    best = (name, r["cost"], sp)
            # write winner
            shutil.copy2(best[2], OUT / f"task{t:03d}.onnx")
            winners.append({"task": t, "winner": best[0], "cost": best[1],
                            "base_cost": base_r["cost"] if base_r else None,
                            "candidates": {k: v[0] for k, v in margins.items()}})

    print(json.dumps({"changed": changed, "winners": winners}, indent=2))
    adopted = [w for w in winners if w["winner"] != "base"]
    by_src = {}
    for w in adopted:
        by_src.setdefault(w["winner"], []).append(w["task"])
    print(f"\nADOPTED {len(adopted)}/{len(changed)} non-base. by source: "
          + json.dumps(by_src), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
