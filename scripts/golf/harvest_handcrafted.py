#!/usr/bin/env python3
"""Harvest scan: artifacts/handcrafted/ vs the current submission.zip base.

Finds strictly-cheaper, correct candidates that campaigns (codex/Fable via
gpt_rebuild.sh -> try_candidate.py) have banked into artifacts/handcrafted/.
Scores each net in a fresh subprocess (ORT contamination isolation, see memory
local-ort-contamination) with an Einsum-hang skip (>=15 operand Einsum) and a
60s timeout. Winners = cheaper-than-base AND locally correct.

Usage: harvest_handcrafted.py            # base = submission.zip, out = /tmp/harvest_handcrafted.json
Then: fresh-gate winners (verify_fix.py --k 30|100), structural-gate, merge, submit.
See docs/minutes/2026-07-03/01-playbook-harvest.md for the full workflow.
"""
from __future__ import annotations
import sys, os, json, math, zipfile, tempfile, multiprocessing as mp
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))
import onnx  # noqa: E402

MAX_EINSUM = 15
TIMEOUT = 60


def hang_prone(data: bytes) -> bool:
    m = onnx.load_model_from_string(data)
    return any(nd.op_type == "Einsum" and len(nd.input) >= MAX_EINSUM
               for nd in m.graph.node)


def _worker(data: bytes, t: int, q) -> None:
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from lib import scoring
        with tempfile.TemporaryDirectory() as wd:
            s = scoring.score_and_verify(onnx.load_model_from_string(data), t,
                                         wd, label="x", require_correct=True)
        q.put(s["cost"] if s else None)
    except Exception:
        q.put(None)


def cost_of(data: bytes, t: int):
    if hang_prone(data):
        return "HANG"
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(data, t, q))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join(5)
        if p.is_alive():
            p.kill()
        return "HANG"
    try:
        return q.get_nowait()
    except Exception:
        return None


def main() -> None:
    base = {}
    with zipfile.ZipFile(REPO / "submission.zip") as z:
        for n in z.namelist():
            b = os.path.basename(n)
            if b.startswith("task") and b.endswith(".onnx"):
                base[int(b[4:7])] = z.read(n)
    diffs = []
    for p in sorted((REPO / "artifacts" / "handcrafted").glob("task*.onnx")):
        t = int(p.name[4:7])
        d = p.read_bytes()
        if t in base and d != base[t]:
            diffs.append((t, p, d))
    out = []
    print(f"diff files: {len(diffs)}", flush=True)
    for t, p, d in diffs:
        bc = cost_of(base[t], t)
        cc = cost_of(d, t)
        row = {"task": t, "base_cost": bc, "cand_cost": cc, "path": str(p)}
        if isinstance(bc, int) and isinstance(cc, int) and cc < bc:
            row["gain"] = round(max(1, 25 - math.log(cc)) - max(1, 25 - math.log(bc)), 4)
        out.append(row)
    winners = [r for r in out if "gain" in r]
    print(f"WINNERS={len(winners)} projected=+{sum(r['gain'] for r in winners):.3f}", flush=True)
    for r in sorted(winners, key=lambda x: -x["gain"]):
        print(f"  task{r['task']:03d} {r['base_cost']}->{r['cand_cost']} +{r['gain']}", flush=True)
    json.dump(out, open("/tmp/harvest_handcrafted.json", "w"), indent=1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
