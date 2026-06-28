#!/usr/bin/env python3
"""For the invalid base tasks, see if others/ holds a grader-valid, correct,
cheaper-or-equal replacement that the auto-harvest skipped (it skips tasks whose
broken base can't be scored). Also report combined_gate_all verdicts.
"""
from __future__ import annotations
import os, io, json, zipfile, hashlib, tempfile, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission.zip"
OTHERS = REPO / "others"
TARGETS = [173, 216, 285, 367]
STAGE = REPO / "artifacts" / "merge_stage"
OUT = REPO / "artifacts" / "_recover_report.txt"

import onnx
import onnxruntime as ort
import sys
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring

BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function"}
log = []
def L(m): log.append(str(m))

def gate(b):
    try:
        m = onnx.load_model_from_string(b)
    except Exception as e:
        return False, f"parse:{e}"
    for n in m.graph.node:
        if n.op_type in BANNED:
            return False, f"banned:{n.op_type}"
    if len(m.graph.sparse_initializer) > 0:
        return False, "sparse"
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ort.InferenceSession(b, sess_options=so, providers=["CPUExecutionProvider"])
    except Exception as e:
        return False, f"ort:{str(e)[:70]}"
    return True, "ok"

def members(p):
    d={}
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            b=os.path.basename(n)
            if b.startswith("task") and b.endswith(".onnx"):
                d[int(b[4:7])]=z.read(n)
    return d

def main():
    base = members(BASE)
    # combined_gate verdicts
    cg = json.load(open(OTHERS/"combined_gate_all.json"))
    cgm = {e["task"]: e for e in cg}
    nok = sum(1 for e in cg if e.get("ok"))
    L(f"combined_gate_all.json: {nok}/{len(cg)} ok")
    for t in TARGETS:
        L(f"  combined_gate task{t:03d}: {cgm.get(t)}")

    # gather all candidate bytes per target from others/ (zips + loose onnx)
    import re
    TR = re.compile(r"task(\d{3})")
    cand = {t: {} for t in TARGETS}
    for zp in sorted(OTHERS.glob("*.zip")):
        try:
            d = members(zp)
        except Exception:
            continue
        for t in TARGETS:
            if t in d:
                cand[t].setdefault(hashlib.sha1(d[t]).hexdigest(), (d[t], zp.name))
    for f in sorted(OTHERS.glob("*.onnx")):
        m = TR.search(f.name)
        if not m: continue
        t = int(m.group(1))
        if t in TARGETS:
            b = f.read_bytes()
            cand[t].setdefault(hashlib.sha1(b).hexdigest(), (b, f.name))

    def sc(c): return max(1.0, 25-math.log(c)) if c and c>0 else 1.0

    recovered = []
    for t in TARGETS:
        bok, bwhy = gate(base[t])
        L(f"\ntask{t:03d}: base gate={bok} ({bwhy[:60]}); {len(cand[t])} distinct candidates in others/")
        best = None
        for sha,(b,src) in cand[t].items():
            ok,why = gate(b)
            if not ok:
                continue
            # grader-safe; now check correctness + cost on local full set
            try:
                r = scoring.score_and_verify(onnx.load_model_from_string(b), t, tempfile.mkdtemp(), label="x", require_correct=False)
            except Exception as e:
                L(f"    {src[:40]}: valid-load but score err {str(e)[:50]}")
                continue
            if r and r.get("correct") and r.get("cost"):
                L(f"    CAND {src[:40]}: VALID+CORRECT cost={r['cost']} score={sc(r['cost']):.3f}")
                if best is None or r["cost"] < best[0]:
                    best = (r["cost"], b, src)
            else:
                L(f"    {src[:40]}: valid-load, correct={r.get('correct') if r else None} cost={r.get('cost') if r else None}")
        if best:
            (STAGE / f"task{t:03d}.onnx").write_bytes(best[1])
            L(f"  >> RECOVER task{t:03d} -> staged valid correct cost={best[0]} (+{sc(best[0]):.3f} vs broken base=0) from {best[2]}")
            recovered.append((t, best[0], sc(best[0]), best[2]))
        else:
            L(f"  -- no valid+correct replacement for task{t:03d} in others/")
    L(f"\nRECOVERED {len(recovered)} tasks: {[r[0] for r in recovered]}")
    L(f"approx point gain (assuming broken base scored 0): +{sum(r[2] for r in recovered):.2f}")
    OUT.write_text("\n".join(log))
    print("done")

if __name__ == "__main__":
    main()
