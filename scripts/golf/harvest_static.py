#!/usr/bin/env python3
"""Re-harvest /others against the current best, costing heavy (giant-Einsum)
candidates by STATIC shape inference only (no execution => no hang).

Base = artifacts/_BEST_7679.81.zip. Incumbent cost = all_scores.csv.
For each candidate (deduped, landmine zips excluded):
  - static cost = params + sum(intermediate tensor bytes) via shape inference.
  - prune if static_cost >= incumbent cost (cannot win).
  - giant-Einsum  -> UNVERIFIED winner (correctness can't be checked locally;
                     grader/A-B must judge).
  - non-giant     -> subprocess verify (require_correct); VERIFIED winner if ok.
Known private-0 culprits never re-adopted: task192, task277, task325.

Outputs winners to /tmp/harvest2_winners/ (verified) and /tmp/harvest2_giants/
(unverified giants) + a JSON report. Does NOT touch submission.zip.
"""
from __future__ import annotations
import sys, os, re, csv, json, zipfile, tempfile
import multiprocessing as mp
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import onnx  # noqa: E402
from onnx import shape_inference, helper  # noqa: E402

OTHERS = REPO / "others"
BASE_ZIP = REPO / "artifacts" / "_BEST_7679.81.zip"
ALL_SCORES = REPO / "all_scores.csv"
WIN_DIR = Path("/tmp/harvest2_winners")
GIANT_DIR = Path("/tmp/harvest2_giants")
REPORT = Path("/tmp/harvest2_report.json")

EXCLUDE_ZIPS = {"submission (18).zip", "submission (20).zip"}
KNOWN_BAD = {192, 277, 325}          # confirmed private-0 / grader-error
MAX_EINSUM_OPERANDS = 12
MAX_EINSUM_EQLEN = 60
VERIFY_TIMEOUT = 25


def is_giant(m: onnx.ModelProto) -> bool:
    for nd in m.graph.node:
        if nd.op_type != "Einsum":
            continue
        if len(nd.input) >= MAX_EINSUM_OPERANDS:
            return True
        for a in nd.attribute:
            if a.name == "equation" and len(a.s) >= MAX_EINSUM_EQLEN:
                return True
    return False


def static_cost(m: onnx.ModelProto) -> int | None:
    try:
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
                    ok = False
                    break
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


def _verify_worker(b: bytes, t: int, q) -> None:
    try:
        from lib import scoring
        with tempfile.TemporaryDirectory() as wd:
            r = scoring.score_and_verify(onnx.load_model_from_string(b), t, wd,
                                         label="hv", require_correct=False)
        if r and r.get("score") is not None:
            q.put((int(r["cost"]), bool(r["correct"])))
        else:
            q.put(None)
    except Exception:
        q.put(None)


def verify(b: bytes, t: int):
    q = mp.Queue()
    p = mp.Process(target=_verify_worker, args=(b, t, q))
    p.start(); p.join(VERIFY_TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join(3)
        if p.is_alive():
            p.kill()
        return "HANG"
    try:
        return q.get_nowait()
    except Exception:
        return None


def base_costs() -> dict[int, int]:
    d = {}
    for r in csv.DictReader(open(ALL_SCORES)):
        try:
            d[int(r["task"].replace("task", ""))] = int(r["cost"])
        except (KeyError, ValueError):
            continue
    return d


def collect() -> dict[int, list[bytes]]:
    pool: dict[int, list[bytes]] = {}
    seen: set[tuple[int, int]] = set()

    def add(t, b):
        h = hash(b)
        if (t, h) in seen:
            return
        seen.add((t, h))
        pool.setdefault(t, []).append(b)

    for op in OTHERS.glob("*.onnx"):
        mo = re.match(r"task0*(\d+)", op.name)
        if mo:
            add(int(mo.group(1)), op.read_bytes())
    for zp in OTHERS.glob("*.zip"):
        if zp.name in EXCLUDE_ZIPS:
            continue
        try:
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    bn = os.path.basename(n)
                    if bn.startswith("task") and bn.endswith(".onnx"):
                        try:
                            add(int(bn[4:7]), z.read(n))
                        except ValueError:
                            pass
        except Exception:
            pass
    for d in OTHERS.glob("submission (*)"):
        if d.is_dir():
            for op in d.glob("task*.onnx"):
                try:
                    add(int(op.name[4:7]), op.read_bytes())
                except ValueError:
                    pass
    return pool


def main() -> int:
    base = base_costs()
    pool = collect()
    for d in (WIN_DIR, GIANT_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.onnx"):
            f.unlink()

    verified, giants = {}, {}
    n_eval = 0
    for t in sorted(pool):
        if t in KNOWN_BAD:
            continue
        inc = base.get(t)
        if inc is None:
            continue
        # static-cost every candidate; keep cheapest that beats incumbent
        cands = []
        for b in pool[t]:
            try:
                m = onnx.load_model_from_string(b)
            except Exception:
                continue
            c = static_cost(m)
            if c is None or c >= inc:
                continue
            cands.append((c, b, is_giant(m)))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0])
        n_eval += 1
        # try cheapest first; verify non-giants, accept giants unverified
        for c, b, giant in cands:
            if giant:
                giants[t] = {"cost": c, "inc": inc}
                (GIANT_DIR / f"task{t:03d}.onnx").write_bytes(b)
                break
            v = verify(b, t)
            if v == "HANG":
                # behaves like a giant at runtime -> static cost, unverified
                giants[t] = {"cost": c, "inc": inc}
                (GIANT_DIR / f"task{t:03d}.onnx").write_bytes(b)
                break
            if v is None:
                continue
            vc, ok = v
            if ok:
                verified[t] = {"cost": vc, "inc": inc}
                (WIN_DIR / f"task{t:03d}.onnx").write_bytes(b)
                break

    import math
    def sc(c): return max(1.0, 25 - math.log(c)) if c > 0 else 25.0
    gain_v = sum(sc(w["cost"]) - sc(w["inc"]) for w in verified.values())
    gain_g = sum(sc(w["cost"]) - sc(w["inc"]) for w in giants.values())
    REPORT.write_text(json.dumps({"verified": {str(k): v for k, v in verified.items()},
                                  "giants": {str(k): v for k, v in giants.items()}}, indent=1))
    print(f"tasks with a cheaper candidate: {n_eval}")
    print(f"\n=== VERIFIED winners (non-giant, public-correct): {len(verified)}  (+{gain_v:.2f}) ===")
    for t in sorted(verified):
        w = verified[t]
        print(f"  task{t:03d}: {w['inc']} -> {w['cost']}")
    print(f"\n=== UNVERIFIED GIANT winners (static cost, grader-judge): {len(giants)}  (+{gain_g:.2f}) ===")
    for t in sorted(giants):
        w = giants[t]
        print(f"  task{t:03d}: {w['inc']} -> {w['cost']} (static)")
    print(f"\nverified -> {WIN_DIR}   giants -> {GIANT_DIR}   report -> {REPORT}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
