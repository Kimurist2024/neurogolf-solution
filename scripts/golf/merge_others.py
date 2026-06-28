#!/usr/bin/env python3
"""Multi-source merge scan for /others.

For every task, scores each source's ONNX (cost + public correctness) and picks
the cheapest CORRECT candidate. Emits, for each task whose best candidate is
cheaper than the current submission's incumbent, the winner (written to
/tmp/merge_winners/taskNNN.onnx) and a manifest. Winners are NOT adopted here —
they must pass verify_fix.py fresh-gate (k=30, <=4 fails) before merging.

Run: uv run python scripts/golf/merge_others.py
Output: /tmp/merge_winners/  (winner onnx files) + /tmp/merge_scan.json (full scan)
        + /tmp/merge_manifest.txt (task=path,... for verify_fix --batch)
"""
from __future__ import annotations
import sys, os, zipfile, tempfile, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import onnx  # noqa: E402
from lib import scoring  # noqa: E402

BASE_ZIP = REPO / "submission.zip"
OTHERS = REPO / "others"
WIN_DIR = Path("/tmp/merge_winners")


def load_zip(path: Path) -> dict[int, bytes]:
    d: dict[int, bytes] = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            b = os.path.basename(n)
            if b.startswith("task") and b.endswith(".onnx"):
                try:
                    t = int(b[4:7])
                except ValueError:
                    continue
                d[t] = z.read(n)
    return d


def _score(args):
    sname, t, b = args
    wd = tempfile.mkdtemp()
    try:
        m = onnx.load_model_from_string(b)
        r = scoring.score_and_verify(m, t, wd, label="m", require_correct=False)
        if r is None:
            return (sname, t, None, False)
        return (sname, t, int(r["cost"]), bool(r["correct"]))
    except Exception:
        return (sname, t, None, False)


def main() -> int:
    sources: dict[str, dict[int, bytes]] = {"BASE": load_zip(BASE_ZIP)}
    for zp in sorted(OTHERS.glob("*.zip")):
        sources[zp.name] = load_zip(zp)
    # standalone task*.onnx files (parse task number from filename prefix)
    import re
    for op in sorted(OTHERS.glob("*.onnx")):
        mobj = re.match(r"task0*(\d+)", op.name)
        if mobj:
            t = int(mobj.group(1))
            sources[op.name] = {t: op.read_bytes()}
    print(f"sources: {list(sources)}")

    jobs = [(sname, t, b) for sname, d in sources.items() for t, b in d.items()]
    print(f"scanning {len(jobs)} (source,task) candidates...")
    results: dict[tuple[str, int], tuple] = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (sname, t, cost, correct) in enumerate(ex.map(_score, jobs, chunksize=8)):
            results[(sname, t)] = (cost, correct)
            if (i + 1) % 400 == 0:
                print(f"  {i+1}/{len(jobs)}")

    WIN_DIR.mkdir(parents=True, exist_ok=True)
    scan = {}
    manifest = []
    adopted_candidates = 0
    for t in range(1, 401):
        base_cost, base_ok = results.get(("BASE", t), (None, False))
        # cheapest correct across ALL sources
        best = None  # (cost, sname)
        for sname in sources:
            cost, ok = results.get((sname, t), (None, False))
            if ok and cost is not None and (best is None or cost < best[0]):
                best = (cost, sname)
        entry = {"base_cost": base_cost, "base_ok": base_ok,
                 "best_cost": best[0] if best else None,
                 "best_src": best[1] if best else None}
        scan[t] = entry
        # winner only if strictly cheaper than base incumbent AND not base itself
        if (best and base_cost is not None and best[0] < base_cost
                and best[1] != "BASE"):
            b = sources[best[1]][t]
            wp = WIN_DIR / f"task{t:03d}.onnx"
            wp.write_bytes(b)
            manifest.append(f"{t}={wp}")
            adopted_candidates += 1
            entry["winner"] = best[1]
            entry["delta"] = best[0] - base_cost

    Path("/tmp/merge_scan.json").write_text(json.dumps(scan, indent=0))
    Path("/tmp/merge_manifest.txt").write_text(",".join(manifest))
    print(f"\nwinners cheaper than base (pre-fresh-gate): {adopted_candidates}")
    for t in range(1, 401):
        e = scan[t]
        if e.get("winner"):
            print(f"  task{t:03d}: {e['base_cost']} -> {e['best_cost']} "
                  f"(Δ{e['delta']}) from {e['winner']}")
    print(f"\nmanifest -> /tmp/merge_manifest.txt  winners -> {WIN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
