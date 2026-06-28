#!/usr/bin/env python3
"""Pool-free sequential harvest scan over a root of mixed sources.

Generalizes scan_zips_seq.py: a "source" is EITHER a *.zip OR a directory that
directly contains taskNNN.onnx files (an unzipped submission). Scores every
candidate sequentially with a per-model signal.alarm timeout (ProcessPoolExecutor
deadlocks with onnxruntime). Skips candidates byte-identical to base; dedups
identical candidates by sha1 across all sources.

Winner = cheapest CORRECT candidate strictly cheaper than the base incumbent.
Output: /tmp/<tag>_winners/ + /tmp/<tag>_winners.json + /tmp/<tag>_scan.log

Run: uv run python scripts/golf/scan_sources_seq.py BEST_ZIP SOURCES_ROOT TAG
"""
from __future__ import annotations
import sys, os, signal, tempfile, zipfile, json, hashlib, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402
import onnx  # noqa: E402

BEST = Path(sys.argv[1])
ROOT = Path(sys.argv[2])
TAG = sys.argv[3] if len(sys.argv) > 3 else "src"
LOG = open(f"/tmp/{TAG}_scan.log", "w")
WIN_DIR = Path(f"/tmp/{TAG}_winners")
PER_MODEL_TIMEOUT = 45


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))


def L(m: str) -> None:
    LOG.write(m + "\n")
    LOG.flush()
    print(m, flush=True)


def _task_of(name: str):
    b = os.path.basename(name)
    if b.startswith("task") and b.endswith(".onnx"):
        try:
            return int(b[4:7])
        except ValueError:
            return None
    return None


def load_zip(p: Path) -> dict[int, bytes]:
    d: dict[int, bytes] = {}
    try:
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                t = _task_of(n)
                if t is not None:
                    d[t] = z.read(n)
    except Exception:
        pass
    return d


def load_dir(p: Path) -> dict[int, bytes]:
    d: dict[int, bytes] = {}
    for f in p.glob("task*.onnx"):
        t = _task_of(f.name)
        if t is not None:
            try:
                d[t] = f.read_bytes()
            except Exception:
                pass
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

    # discover sources: all zips + all dirs that directly hold task*.onnx
    zips = sorted(ROOT.rglob("*.zip"))
    dir_set = set()
    for f in ROOT.rglob("task*.onnx"):
        if _task_of(f.name) is not None:
            dir_set.add(f.parent)
    dirs = sorted(dir_set)
    L(f"sources under {ROOT}: {len(zips)} zips, {len(dirs)} onnx-dirs")

    cands: dict[int, dict[str, tuple[bytes, str]]] = {}
    n_raw = 0
    for src, loader, label in ([(z, load_zip, z.name) for z in zips] +
                               [(d, load_dir, str(d.relative_to(ROOT))) for d in dirs]):
        data = loader(src)
        for t, b in data.items():
            n_raw += 1
            if t in base and b == base[t]:
                continue
            sha = hashlib.sha1(b).hexdigest()
            slot = cands.setdefault(t, {})
            if sha not in slot:
                slot[sha] = (b, label)

    tasks = sorted(cands)
    ncand = sum(len(v) for v in cands.values())
    L(f"raw candidates: {n_raw}; tasks with differing candidates: {len(tasks)}; unique: {ncand}")

    WIN_DIR.mkdir(parents=True, exist_ok=True)
    winners = []
    t0 = time.time()
    for i, t in enumerate(tasks):
        br = score(base[t], t) if t in base else None
        bc = br["cost"] if br else None
        best = None
        for sha, (b, label) in cands[t].items():
            r = score(b, t)
            if r and r.get("correct") and r.get("cost") is not None:
                if best is None or r["cost"] < best[0]:
                    best = (r["cost"], label, b)
        if best and bc is not None and best[0] < bc:
            wp = WIN_DIR / f"task{t:03d}.onnx"
            wp.write_bytes(best[2])
            winners.append({"task": t, "base": bc, "cand": best[0],
                            "src": best[1], "path": str(wp)})
            L(f"  WIN task{t:03d} {bc}->{best[0]} (Δ{best[0]-bc}) from {best[1]}")
        if (i + 1) % 25 == 0:
            L(f"  ...{i+1}/{len(tasks)} ({time.time()-t0:.0f}s)")

    json.dump(winners, open(f"/tmp/{TAG}_winners.json", "w"), indent=0)
    L(f"DONE {TAG} winners={len(winners)} ({time.time()-t0:.0f}s) -> /tmp/{TAG}_winners.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
