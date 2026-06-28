#!/usr/bin/env python3
"""Pool-free sequential scan of others/*.zip against the current best.

ProcessPoolExecutor + onnxruntime deadlocks (see merge_others.py orphans), so
this scores every candidate sequentially with a per-model signal.alarm timeout.
Skips candidates byte-identical to base; dedups identical candidates by sha1.

Winner = cheapest CORRECT candidate strictly cheaper than the base incumbent.
Winners (NOT auto-adopted) -> /tmp/zip_winners/ + /tmp/zip_winners.json; they
must still pass verify_fix.py fresh-gate (k=30) before merge.

Run: uv run python scripts/golf/scan_zips_seq.py [BEST_ZIP]
"""
from __future__ import annotations
import sys, os, signal, tempfile, zipfile, json, hashlib, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402
import onnx  # noqa: E402

BEST = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "artifacts/_BEST_7398.65.zip"
OTHERS = REPO / "others"
WIN_DIR = Path("/tmp/zip_winners")
LOG = open("/tmp/zipscan.log", "w")
PER_MODEL_TIMEOUT = 45


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))


def L(m: str) -> None:
    LOG.write(m + "\n")
    LOG.flush()
    print(m, flush=True)


def load_zip(p: Path) -> dict[int, bytes]:
    d: dict[int, bytes] = {}
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            b = os.path.basename(n)
            if b.startswith("task") and b.endswith(".onnx"):
                try:
                    t = int(b[4:7])
                except ValueError:
                    continue
                d[t] = z.read(n)
    return d


def score(b: bytes, t: int):
    signal.alarm(PER_MODEL_TIMEOUT)
    try:
        r = scoring.score_and_verify(
            onnx.load_model_from_string(b), t, tempfile.mkdtemp(),
            label="x", require_correct=False)
        signal.alarm(0)
        return r
    except Exception:
        signal.alarm(0)
        return None


def main() -> int:
    base = load_zip(BEST)
    L(f"base {BEST.name}: {len(base)} tasks")
    zips = sorted(OTHERS.glob("*.zip"))
    L(f"zips: {[z.name for z in zips]}")

    # candidates differing from base, deduped by sha1
    cands: dict[int, dict[str, tuple[bytes, list[str]]]] = {}
    for zp in zips:
        d = load_zip(zp)
        for t, b in d.items():
            if t in base and b == base[t]:
                continue
            sha = hashlib.sha1(b).hexdigest()
            slot = cands.setdefault(t, {})
            if sha in slot:
                slot[sha][1].append(zp.name)
            else:
                slot[sha] = (b, [zp.name])

    tasks = sorted(cands)
    ncand = sum(len(v) for v in cands.values())
    L(f"tasks with differing candidates: {len(tasks)}; unique candidates: {ncand}")

    WIN_DIR.mkdir(parents=True, exist_ok=True)
    winners = []
    t0 = time.time()
    for i, t in enumerate(tasks):
        br = score(base[t], t) if t in base else None
        bc = br["cost"] if br else None
        best = None  # (cost, src, bytes)
        for sha, (b, srcs) in cands[t].items():
            r = score(b, t)
            if r and r.get("correct") and r.get("cost") is not None:
                if best is None or r["cost"] < best[0]:
                    best = (r["cost"], srcs[0], b)
        if best and bc is not None and best[0] < bc:
            wp = WIN_DIR / f"task{t:03d}.onnx"
            wp.write_bytes(best[2])
            winners.append({"task": t, "base": bc, "cand": best[0],
                            "src": best[1], "path": str(wp)})
            L(f"  WIN task{t:03d} {bc}->{best[0]} (Δ{best[0]-bc}) from {best[1]}")
        if (i + 1) % 20 == 0:
            L(f"  ...{i+1}/{len(tasks)} ({time.time()-t0:.0f}s)")

    json.dump(winners, open("/tmp/zip_winners.json", "w"), indent=0)
    L(f"DONE winners={len(winners)} ({time.time()-t0:.0f}s) -> /tmp/zip_winners.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
