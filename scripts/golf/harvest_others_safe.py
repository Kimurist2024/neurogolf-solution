#!/usr/bin/env python3
"""Hang-safe multi-source harvest over /others.

Pools candidate ONNX from every source in /others (loose task*.onnx, the safe
*.zip bundles, and the submission(21) directory), pre-filters out giant-Einsum
nets that hang ONNX Runtime locally (deferred, not scored), then scores the rest
in a process pool. For each task it picks the cheapest PUBLIC-CORRECT candidate
that is strictly cheaper than the current incumbent (cost from all_scores.csv).

Winners are written to /tmp/harvest_winners/ and a manifest; they are NOT adopted
here — they must still pass the fresh k=30 gate before merging.

Landmines excluded: submission (18)/(20).zip (grader-0 incident).
Deferred (hang) tasks are listed so they can be handled separately.

Run: .venv/bin/python3 scripts/golf/harvest_others_safe.py
"""
from __future__ import annotations
import sys, os, re, csv, json, time, hashlib, zipfile, tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FTimeout

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import onnx  # noqa: E402
from lib import scoring  # noqa: E402

OTHERS = REPO / "others"
BASE_ZIP = REPO / "artifacts" / "_BEST_7655.12.zip"
ALL_SCORES = REPO / "all_scores.csv"
WIN_DIR = Path("/tmp/harvest_winners")
SCAN_OUT = Path("/tmp/harvest_scan.json")

# grader-0 landmine bundles — never pool these (see memory: sub18-grader-failure)
EXCLUDE_ZIPS = {"submission (18).zip", "submission (20).zip"}
# giant-Einsum thresholds that hang ORT locally (tightened to catch sub-threshold
# slow nets that previously stalled the ordered pool)
MAX_EINSUM_OPERANDS = 12
MAX_EINSUM_EQLEN = 60
GLOBAL_BUDGET = 600  # seconds; bail + kill workers, take partial results


def inspect(b: bytes):
    """Return (giant_einsum: bool, param_elems: int|None). param_elems is a lower
    bound on cost (cost = params + memory_bytes >= params)."""
    try:
        m = onnx.load_model_from_string(b)
    except Exception:
        return True, None
    for nd in m.graph.node:
        if nd.op_type != "Einsum":
            continue
        if len(nd.input) >= MAX_EINSUM_OPERANDS:
            return True, None
        for a in nd.attribute:
            if a.name == "equation" and len(a.s) >= MAX_EINSUM_EQLEN:
                return True, None
    params = 0
    for init in m.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        params += n
    return False, params


def base_costs() -> dict[int, int]:
    d: dict[int, int] = {}
    for r in csv.DictReader(open(ALL_SCORES)):
        try:
            d[int(r["task"].replace("task", ""))] = int(r["cost"])
        except (KeyError, ValueError):
            continue
    return d


def _score(args):
    sname, t, b = args
    wd = tempfile.mkdtemp()
    try:
        m = onnx.load_model_from_string(b)
        r = scoring.score_and_verify(m, t, wd, label="h", require_correct=False)
        if r is None:
            return (sname, t, None, False)
        return (sname, t, int(r["cost"]), bool(r["correct"]))
    except Exception:
        return (sname, t, None, False)


def collect_sources() -> dict[str, dict[int, bytes]]:
    sources: dict[str, dict[int, bytes]] = {}

    def put(src, t, b):
        sources.setdefault(src, {})[t] = b

    # loose task*.onnx
    for op in sorted(OTHERS.glob("*.onnx")):
        mo = re.match(r"task0*(\d+)", op.name)
        if mo:
            put(op.name, int(mo.group(1)), op.read_bytes())
    # safe zips
    for zp in sorted(OTHERS.glob("*.zip")):
        if zp.name in EXCLUDE_ZIPS:
            continue
        try:
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    bn = os.path.basename(n)
                    if bn.startswith("task") and bn.endswith(".onnx"):
                        try:
                            t = int(bn[4:7])
                        except ValueError:
                            continue
                        put(zp.name, t, z.read(n))
        except Exception:
            pass
    # submission(21) unpacked dir
    for d in OTHERS.glob("submission (*)"):
        if d.is_dir():
            for op in d.glob("task*.onnx"):
                try:
                    t = int(op.name[4:7])
                except ValueError:
                    continue
                put(d.name, t, op.read_bytes())
    return sources


def main() -> int:
    sources = collect_sources()
    base = base_costs()
    print(f"sources: {len(sources)}  (excluded: {sorted(EXCLUDE_ZIPS)})")

    jobs = []
    seen_hash: set[tuple[int, str]] = set()  # (task, sha) dedup across sources
    deferred: set[int] = set()
    deferred_srcs: dict[int, list[str]] = {}
    n_cand = pruned_dup = pruned_cost = 0
    for sname, d in sources.items():
        for t, b in d.items():
            n_cand += 1
            bc = base.get(t)
            giant, params = inspect(b)
            if giant:
                deferred.add(t)
                deferred_srcs.setdefault(t, []).append(sname)
                continue
            sha = hashlib.sha256(b).hexdigest()[:16]
            if (t, sha) in seen_hash:
                pruned_dup += 1
                continue
            seen_hash.add((t, sha))
            # param lower-bound prune: cost >= params, so can't beat incumbent
            if bc is not None and params is not None and params >= bc:
                pruned_cost += 1
                continue
            jobs.append((sname, t, b))
    print(f"candidates: {n_cand}  to-score: {len(jobs)}  "
          f"(dedup -{pruned_dup}, cost-prune -{pruned_cost})  "
          f"deferred(giant-einsum): {len(deferred)} tasks", flush=True)

    best: dict[int, tuple[int, str]] = {}  # task -> (cost, src) cheapest correct
    done = 0
    t0 = time.time()
    ex = ProcessPoolExecutor(max_workers=8)
    futs = {ex.submit(_score, j): j for j in jobs}
    try:
        for fut in as_completed(futs, timeout=GLOBAL_BUDGET):
            sname, t, cost, correct = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  scored {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
            if correct and cost is not None:
                if t not in best or cost < best[t][0]:
                    best[t] = (cost, sname)
    except FTimeout:
        print(f"  GLOBAL_BUDGET {GLOBAL_BUDGET}s hit at {done}/{len(jobs)} — "
              f"taking partial, killing stuck workers", flush=True)
    # force-kill any lingering (hung) workers so we never deadlock
    for p in list(getattr(ex, "_processes", {}).values()):
        try:
            p.kill()
        except Exception:
            pass
    ex.shutdown(wait=False, cancel_futures=True)

    WIN_DIR.mkdir(parents=True, exist_ok=True)
    for f in WIN_DIR.glob("*.onnx"):
        f.unlink()
    winners = {}
    manifest = []
    for t, (cost, sname) in sorted(best.items()):
        bc = base.get(t)
        if bc is not None and cost < bc:
            (WIN_DIR / f"task{t:03d}.onnx").write_bytes(sources[sname][t])
            winners[t] = {"cost": cost, "src": sname, "base": bc, "delta": cost - bc}
            manifest.append(f"{t}={WIN_DIR / f'task{t:03d}.onnx'}")

    SCAN_OUT.write_text(json.dumps(
        {"winners": {str(k): v for k, v in winners.items()},
         "deferred": sorted(deferred),
         "deferred_srcs": {str(k): v for k, v in deferred_srcs.items()}}, indent=1))
    Path("/tmp/harvest_manifest.txt").write_text(",".join(manifest))

    print(f"\n=== WINNERS cheaper than incumbent (pre fresh-gate): {len(winners)} ===")
    tot = 0
    for t in sorted(winners):
        w = winners[t]
        tot += -w["delta"]
        print(f"  task{t:03d}: {w['base']} -> {w['cost']}  (Δ{w['delta']})  [{w['src']}]")
    print(f"  total cost reduction: {tot}")
    print(f"\n=== DEFERRED (giant-Einsum / hang, NOT processed): {len(deferred)} tasks ===")
    print("  " + " ".join(f"{t:03d}" for t in sorted(deferred)))
    print(f"\nwinners -> {WIN_DIR}   scan -> {SCAN_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
