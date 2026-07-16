#!/usr/bin/env python3
"""Fast 2-phase scan of others/7504 (34k onnx) against current best (all_scores.csv).

Phase1: parallel static_cost, keep candidates strictly cheaper than incumbent.
Phase2: per task, subprocess-verify cheapest-first (isolated, timeout), first
        correct & cheaper -> winner. Outputs /tmp/h7504_winners.json + copies.
"""
from __future__ import annotations
import sys, os, re, csv, json, tempfile
import multiprocessing as mp
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import onnx  # noqa: E402
from onnx import shape_inference, helper  # noqa: E402

SRC = REPO / "others" / "7504"
ALL_SCORES = REPO / "all_scores.csv"
WIN_DIR = Path("/tmp/h7504_winners")
OUT = Path("/tmp/h7504_winners.json")
VERIFY_TIMEOUT = 25
MAX_VERIFY_PER_TASK = 8   # cheapest-N correctness attempts per task


def static_cost_of_path(p: str):
    try:
        m = onnx.load_model(p)
        params = sum(int(np.prod(i.dims)) for i in m.graph.initializer)
        g = shape_inference.infer_shapes(m, strict_mode=False).graph
        init = {i.name for i in g.initializer}
        mem = 0
        for vi in list(g.value_info) + list(g.output):
            if vi.name in init or vi.name in ("input", "output"):
                continue
            tt = vi.type.tensor_type
            if not tt.HasField("shape"):
                continue
            n, ok = 1, True
            for d in tt.shape.dim:
                if not d.HasField("dim_value") or d.dim_value <= 0:
                    ok = False; break
                n *= d.dim_value
            if ok:
                try:
                    it = np.dtype(helper.tensor_dtype_to_np_dtype(tt.elem_type)).itemsize
                except Exception:
                    it = 4
                mem += n * it
        return params + mem
    except Exception:
        return None


def _sc_worker(p):
    return (p, static_cost_of_path(p))


def _verify_worker(b, t, q):
    try:
        from lib import scoring
        with tempfile.TemporaryDirectory() as wd:
            r = scoring.score_and_verify(onnx.load_model_from_string(b), t, wd,
                                         label="hv", require_correct=False)
        q.put((int(r["cost"]), bool(r["correct"])) if r and r.get("score") is not None else None)
    except Exception:
        q.put(None)


def verify(b, t):
    q = mp.Queue()
    p = mp.Process(target=_verify_worker, args=(b, t, q))
    p.start(); p.join(VERIFY_TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join(3)
        if p.is_alive(): p.kill()
        return "HANG"
    try:
        return q.get_nowait()
    except Exception:
        return None


def base_costs():
    d = {}
    for r in csv.DictReader(open(ALL_SCORES)):
        try:
            d[int(r["task"].replace("task", ""))] = int(r["cost"])
        except (KeyError, ValueError):
            continue
    return d


def L(m):
    print(m, flush=True)


def main():
    inc = base_costs()
    # collect files by task
    files = list(SRC.rglob("*.onnx"))
    L(f"files: {len(files)}")
    # Phase1: parallel static cost
    with mp.Pool(min(8, os.cpu_count() or 4)) as pool:
        results = pool.map(_sc_worker, [str(f) for f in files], chunksize=64)
    # group survivors (static_cost < incumbent) by task, dedupe by (task,cost,size)
    bytask = {}
    for p, sc in results:
        if sc is None:
            continue
        mo = re.search(r"task(\d{3})", os.path.basename(p))
        if not mo:
            continue
        t = int(mo.group(1))
        if t not in inc or sc >= inc[t]:
            continue
        bytask.setdefault(t, []).append((sc, p))
    nsurv = sum(len(v) for v in bytask.values())
    L(f"phase1 survivors (static<incumbent): {nsurv} across {len(bytask)} tasks")
    # Phase2: verify cheapest-first per task
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    winners = []
    for i, t in enumerate(sorted(bytask)):
        cands = sorted(set(bytask[t]))  # by static cost asc, dedup exact (cost,path)
        # dedupe by bytes to avoid re-verifying identical
        seen = set(); uniq = []
        for sc, p in cands:
            b = Path(p).read_bytes()
            h = hash(b)
            if h in seen:
                continue
            seen.add(h); uniq.append((sc, p, b))
        best = None
        for k, (sc, p, b) in enumerate(uniq[:MAX_VERIFY_PER_TASK]):
            r = verify(b, t)
            if r in (None, "HANG"):
                continue
            rc, correct = r
            if correct and rc < inc[t]:
                best = (rc, p, b); break
        if best:
            wp = WIN_DIR / f"task{t:03d}.onnx"
            wp.write_bytes(best[2])
            winners.append({"task": t, "base": inc[t], "cand": best[0], "path": str(wp)})
            L(f"  WIN task{t:03d} {inc[t]}->{best[0]} (Δ{best[0]-inc[t]})")
        if (i + 1) % 25 == 0:
            L(f"  ...{i+1}/{len(bytask)}")
    json.dump(winners, open(OUT, "w"), indent=0)
    tot = sum(np.log(w["base"]) - np.log(w["cand"]) for w in winners)
    L(f"DONE h7504 winners={len(winners)} projected_gain=+{tot:.4f} -> {OUT}")


if __name__ == "__main__":
    main()
