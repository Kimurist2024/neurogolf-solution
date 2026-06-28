#!/usr/bin/env python3
"""Autonomous harvest loop for the Codex rebuild campaign.

Polls artifacts/handcrafted/ for the bottom-20 targets. For each net now cheaper
than the reference (/tmp/hc_best.json), fresh-gates it (verify_fix --k 30); if
ADOPT, adds to /tmp/pending_harvest.json and bumps the reference. Stays SILENT
during routine accumulation. Emits a line (= a chat notification) ONLY on:
  BIG task<NNN>  ...     a single task gained >= 1.0 (monster rebuild cracked)
  READY_SUBMIT total+X   cumulative pending gain >= THRESH -> then EXITS
  CAMPAIGN_ENDED ...     daemon + workers gone -> emit remaining pending, EXIT

The MAIN session submits the pending batch on READY_SUBMIT / CAMPAIGN_ENDED,
then relaunches this loop (with a refreshed reference) to keep harvesting.
"""
from __future__ import annotations
import sys, os, json, math, time, signal, tempfile, subprocess, csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402
import onnx  # noqa: E402

HAND = REPO / "artifacts" / "handcrafted"
BEST = Path("/tmp/hc_best.json")
PEND = Path("/tmp/pending_harvest.json")
THRESH = float(open("/tmp/harvest_threshold.txt").read().strip()) if Path("/tmp/harvest_threshold.txt").exists() else 4.0
VENV = str(REPO / ".venv/bin/python")
ORIG = {int(r["task"][4:]): int(r["cost"]) for r in csv.DictReader(open("all_scores.csv"))}


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))


def score(p, t):
    signal.alarm(40)
    try:
        r = scoring.score_and_verify(onnx.load(str(p)), t, tempfile.mkdtemp(), label="x", require_correct=False)
        signal.alarm(0)
        return r
    except Exception:
        signal.alarm(0)
        return None


def gate(t, path):
    try:
        out = subprocess.run([VENV, "scripts/verify_fix.py", "--batch", f"{t}={path}", "--k", "30"],
                             capture_output=True, text=True, timeout=600).stdout
        return f"task{t:03d}: ADOPT" in out
    except Exception:
        return False


def total(pend):
    return sum(math.log(ORIG[int(k)]) - math.log(v["cost"]) for k, v in pend.items() if int(k) in ORIG)


def alive():
    for pat in ("gpt_rebuild.sh", "codex exec"):
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
            return True
    return False


def main():
    while True:
        best = json.load(open(BEST))
        pend = json.load(open(PEND)) if PEND.exists() else {}
        for ts, bc in list(best.items()):
            t = int(ts)
            p = HAND / f"task{t:03d}.onnx"
            if not p.is_file():
                continue
            r = score(p, t)
            if not r or not r.get("correct") or r.get("cost") is None:
                continue
            c = int(r["cost"])
            if c < int(bc):
                if gate(t, str(p)):
                    pend[ts] = {"cost": c, "path": str(p)}
                    json.dump(pend, open(PEND, "w"))
                    g = math.log(ORIG[t]) - math.log(c) if t in ORIG else 0.0
                    if g >= 1.0:
                        print(f"BIG task{t:03d} {ORIG.get(t)}->{c} gain+{g:.3f}", flush=True)
                best[ts] = c  # acknowledge (adopt or reject) to avoid re-gating same cost
                json.dump(best, open(BEST, "w"))
        tot = total(pend)
        if tot >= THRESH:
            print(f"READY_SUBMIT total+{tot:.3f} tasks={sorted(int(k) for k in pend)}", flush=True)
            return
        if not alive():
            print(f"CAMPAIGN_ENDED pending+{tot:.3f} tasks={sorted(int(k) for k in pend)}", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
