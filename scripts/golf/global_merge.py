#!/usr/bin/env python3
"""Global best-net merge into the solo stage.

For every task, if artifacts/handcrafted/taskXXX.onnx is cheaper than the current
solo stage net AND passes the strict fresh-gate (verify_fix, k=SOLO_K), adopt it
into artifacts/golf_solo/stage. This pulls the kimi/codex lane's accumulated
handcrafted gains into the submission, with the same fresh-0-fail + dual-gold +
margin guarantee solo uses (overfit nets are rejected).

Run with BOTH loops paused so handcrafted is a stable snapshot. Updates
solo_state.json (champion_cost, projected, and PROMOTED status for tracked
targets that cross THR), rebuilds artifacts/golf_solo/submission.zip, and prints
a summary. Submission + bump-submit are done by the caller.

Usage: global_merge.py [--k 500]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import onnx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402
import verify_fix  # noqa: E402

STAGE = REPO / "artifacts" / "golf_solo" / "stage"
HC = REPO / "artifacts" / "handcrafted"
SUB_ZIP = REPO / "artifacts" / "golf_solo" / "submission.zip"
STATE = REPO / "docs" / "golf" / "solo_state.json"
LOG_MD = REPO / "docs" / "golf" / "solo_improvements.md"
SIZE_LIMIT = 1_509_949

THR = float(os.environ.get("SOLO_THR", "15.1"))


def sc(c: float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(c))))


def _hc_cost(task: int) -> tuple[int, int | None]:
    """Cost-only profile of a handcrafted net (fast pre-filter)."""
    p = HC / f"task{task:03d}.onnx"
    if not p.is_file():
        return task, None
    with tempfile.TemporaryDirectory() as wd:
        try:
            r = scoring.score_and_verify(onnx.load(str(p)), task, wd, label="m",
                                         require_correct=False)
        except Exception:
            return task, None
    return task, (int(r["cost"]) if r else None)


def _fresh_gate(args: tuple[int, int]) -> dict:
    """Strict fresh-gate one candidate task (runs in a worker process)."""
    task, k = args
    v = verify_fix.verify_one(task, HC / f"task{task:03d}.onnx", k)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=int(os.environ.get("SOLO_K", "500")))
    a = ap.parse_args()

    st = json.loads(STATE.read_text())
    champ = st["champion_cost"]              # current stage cost per task (all 400)
    workers = max(1, (os.cpu_count() or 4) - 1)

    # 1) fast cost-only pre-filter: which handcrafted nets are cheaper than stage?
    tasks = list(range(1, 401))
    cand = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for task, hc in ex.map(_hc_cost, tasks):
            if hc is not None and hc < champ[str(task)]:
                cand.append((task, hc, champ[str(task)]))
    cand.sort(key=lambda x: sc(x[1]) - sc(x[2]), reverse=True)  # biggest gain first
    print(f"pre-filter: {len(cand)} handcrafted nets cheaper than stage "
          f"(optimistic gain +{sum(sc(c)-sc(s) for _,c,s in cand):.2f})")

    if not cand:
        print("nothing to merge.")
        return 0

    # 2) strict fresh-gate each candidate (k); adopt only ADOPT + cheaper.
    adopted, gain, rejected = [], 0.0, []
    fg_workers = max(1, min(workers, len(cand)))
    with ProcessPoolExecutor(max_workers=fg_workers) as ex:
        verdicts = list(ex.map(_fresh_gate, [(t, a.k) for t, _, _ in cand]))

    for (task, hc, old), v in zip(cand, verdicts):
        ok = (v.get("decision") == "ADOPT" and v.get("cost") is not None
              and v["cost"] < old)
        if ok:
            new = int(v["cost"])
            shutil.copy2(HC / f"task{task:03d}.onnx", STAGE / f"task{task:03d}.onnx")
            gain += sc(new) - sc(old)
            champ[str(task)] = new
            adopted.append((task, old, new))
            # if a tracked solo target crosses THR, mark it PROMOTED.
            ts = st["task_state"].get(str(task))
            if ts is not None:
                ts["best_cost"] = new
                if sc(new) >= st["thr"] and ts["status"] != "PROMOTED":
                    ts["status"] = "PROMOTED"
        else:
            rejected.append((task, v.get("fresh_fails"), v.get("fresh_total"),
                             v.get("official_gold"), v.get("cost")))

    st["projected"] = round(st["projected"] + gain, 4)
    st["adopted_total"] = st.get("adopted_total", 0) + len(adopted)
    STATE.write_text(json.dumps(st, indent=2) + "\n")

    # 3) record + rebuild zip
    with LOG_MD.open("a") as fh:
        for task, old, new in adopted:
            fh.write(f"| {task:03d} | MERGE | {old} | {new} | {sc(old):.3f} | "
                     f"{sc(new):.3f} | {sc(new)-sc(old):+.3f} | - | {a.k} | "
                     f"handcrafted lane (fresh-gated) |\n")

    files = sorted(STAGE.glob("task*.onnx"))
    assert len(files) == 400, f"expected 400, got {len(files)}"
    assert not [f for f in files if f.stat().st_size > SIZE_LIMIT]
    if SUB_ZIP.exists():
        SUB_ZIP.unlink()
    with zipfile.ZipFile(SUB_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)

    print(f"\nADOPTED {len(adopted)} fresh-passing nets, gain +{gain:.3f}")
    for task, old, new in sorted(adopted, key=lambda x: sc(x[1]) - sc(x[2])):
        print(f"  task{task:03d}: {old} -> {new}  (+{sc(new)-sc(old):.3f})")
    if rejected:
        print(f"\nREJECTED {len(rejected)} (failed fresh-gate, NOT merged):")
        for task, ff, ft, og, c in rejected[:20]:
            print(f"  task{task:03d}: fresh_fails={ff}/{ft} official_gold={og} cost={c}")
    print(f"\nnew projected = {st['projected']}  zip={SUB_ZIP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
