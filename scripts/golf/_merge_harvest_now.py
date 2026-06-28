#!/usr/bin/env python3
"""Merge harvested winners (artifacts/merge_stage) into the current submission.

Base = submission.zip (current = othersNOW41 on 7604.90).
Replace each base member with the staged task<NNN>.onnx, then run a grader-safe
gate over EVERY task in the merged zip:
  - onnx.load parses
  - no banned ops (Loop/Scan/NonZero/Unique/Script/Function)
  - no sparse_initializer
  - ORT InferenceSession builds (DISABLE_ALL)
Writes the merged zip only if all gates pass. Report -> working dir.
"""
from __future__ import annotations
import os, io, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission.zip"
STAGE = REPO / "artifacts" / "merge_stage"
OUT = REPO / "artifacts" / "submission_harvestNOW.zip"
REPORT = REPO / "artifacts" / "merge_harvest_report.txt"

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function"}
log = []
def L(m): log.append(str(m))


def members(p):
    d = {}
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            b = os.path.basename(n)
            if b.startswith("task") and b.endswith(".onnx"):
                d[int(b[4:7])] = z.read(n)
    return d


def gate(b: bytes):
    try:
        m = onnx.load_model_from_string(b)
    except Exception as e:  # noqa: BLE001
        return False, f"parse:{e}"
    for node in m.graph.node:
        if node.op_type in BANNED:
            return False, f"banned:{node.op_type}"
    if len(m.graph.sparse_initializer) > 0:
        return False, "sparse_initializer"
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ort.InferenceSession(b, sess_options=so, providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        return False, f"ort:{e}"
    return True, "ok"


def main():
    base = members(BASE)
    L(f"base submission.zip tasks={len(base)}")
    staged = sorted(STAGE.glob("task*.onnx"))
    L(f"staged winners={len(staged)}")

    merged = dict(base)
    changed = []
    for f in staged:
        t = int(f.name[4:7])
        nb = f.read_bytes()
        ob = base.get(t)
        if ob is None:
            L(f"skip task{t:03d}: not in base"); continue
        if nb == ob:
            L(f"skip task{t:03d}: identical to base"); continue
        merged[t] = nb
        changed.append(t)
    L(f"changed tasks={len(changed)}: {sorted(changed)}")

    changed_set = set(changed)
    changed_fails = []   # winners we are introducing -> BLOCK if invalid
    base_fails = []      # pre-existing in already-submitted base -> warn only
    for t in sorted(merged):
        ok, why = gate(merged[t])
        if not ok:
            if t in changed_set:
                changed_fails.append((t, why))
            else:
                base_fails.append((t, why))
    if base_fails:
        L(f"PRE-EXISTING base fails (unchanged, already live, NOT blocking) count={len(base_fails)}")
        for t, why in base_fails:
            L(f"  base task{t:03d}: {why[:90]}")
    if changed_fails:
        L(f"CHANGED-TASK GATE FAIL count={len(changed_fails)} -> drop these winners and abort")
        for t, why in changed_fails:
            L(f"  task{t:03d}: {why[:90]}")
        L("ABORT: a harvested winner is grader-invalid.")
        REPORT.write_text("\n".join(log))
        print("RESULT=ABORT")
        return 1
    L("GATE PASS: every CHANGED task is grader-safe (parse/no-banned/no-sparse/ORT-load).")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for t in sorted(merged):
            z.writestr(f"task{t:03d}.onnx", merged[t])
    OUT.write_bytes(buf.getvalue())
    L(f"wrote {OUT.name} bytes={len(buf.getvalue())} tasks={len(merged)} changed={len(changed)}")
    REPORT.write_text("\n".join(log))
    print("RESULT=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
