#!/usr/bin/env python3
"""Sequential harvest scan of others/a (zips + loose onnx) vs the current best.

Adapted from scan_zips_seq.py / multi_merge_scan.py. Unlike scan_zips_seq, this
covers a SUBFOLDER and LOOSE onnx files, since others/a/ holds 39 submission
zips plus 48 loose task<NNN>*.onnx harvested from teammates/forum.

Sources scanned:
  - every *.zip in others/a   (task<NNN>.onnx members)
  - every loose *.onnx whose name contains task<NNN>

Per task, winner = cheapest CORRECT candidate strictly cheaper than the base
incumbent. Winners are STAGED to artifacts/merge_stage/task<NNN>.onnx + a
manifest.json in the schema merge_finalize.py expects. NOTHING is adopted here;
run merge_finalize.py next to fresh-gate (k=30) and build the merged zip.

ProcessPool + onnxruntime deadlocks, so every candidate is scored sequentially
with a per-model signal.alarm timeout. Identical candidates dedup by sha1.

Run: uv run python scripts/golf/scan_others_a.py [BASE_ZIP]
"""
from __future__ import annotations
import sys, os, signal, tempfile, zipfile, json, hashlib, time, math, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402
import onnx  # noqa: E402

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "artifacts/_BEST_7554.38.zip"
SRC = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "others" / "a"
STAGE = REPO / "artifacts" / "merge_stage"
PER_MODEL_TIMEOUT = 45
TASK_RE = re.compile(r"task(\d{3})")


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))


def L(m: str) -> None:
    print(m, flush=True)


def score_fn(c) -> float:
    return max(1.0, 25 - math.log(c)) if c and c > 0 else 1.0


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
    base = load_zip(BASE)
    L(f"base {BASE.name}: {len(base)} tasks")

    cands: dict[int, dict[str, tuple[bytes, list[str]]]] = {}

    def add(t: int, b: bytes, src: str) -> None:
        if t in base and b == base[t]:
            return
        sha = hashlib.sha1(b).hexdigest()
        slot = cands.setdefault(t, {})
        if sha in slot:
            slot[sha][1].append(src)
        else:
            slot[sha] = (b, [src])

    zips = sorted(SRC.glob("*.zip"))
    L(f"zips: {len(zips)}")
    for zp in zips:
        try:
            d = load_zip(zp)
        except Exception as e:  # noqa: BLE001
            L(f"  bad zip {zp.name}: {e}")
            continue
        for t, b in d.items():
            add(t, b, zp.name)

    loose = sorted(SRC.glob("*.onnx"))
    unmapped: list[str] = []
    unloadable: list[str] = []
    for f in loose:
        m = TASK_RE.search(f.name)
        if not m:
            unmapped.append(f.name)
            continue
        t = int(m.group(1))
        try:
            b = f.read_bytes()
            onnx.load_model_from_string(b)
        except Exception:  # noqa: BLE001
            unloadable.append(f.name)
            continue
        add(t, b, f.name)
    L(f"loose onnx: {len(loose)}  unmapped(no task NNN): {unmapped}  unloadable: {unloadable}")

    tasks = sorted(cands)
    ncand = sum(len(v) for v in cands.values())
    L(f"tasks with differing candidates: {len(tasks)}; unique candidates: {ncand}")

    if STAGE.exists():
        for old in STAGE.glob("*"):
            old.unlink()
    STAGE.mkdir(parents=True, exist_ok=True)

    wins = []
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
            wc = best[0]
            (STAGE / f"task{t:03d}.onnx").write_bytes(best[2])
            d = score_fn(wc) - score_fn(bc)
            wins.append({"task": t, "best": bc, "win_cost": wc,
                         "source": best[1], "delta": round(d, 4)})
            L(f"  WIN task{t:03d} {bc}->{wc} (Δcost {wc-bc}, +{d:.3f}) from {best[1]}")
        if (i + 1) % 20 == 0:
            L(f"  ...{i+1}/{len(tasks)} ({time.time()-t0:.0f}s)")

    wins.sort(key=lambda r: -r["delta"])
    (STAGE / "manifest.json").write_text(json.dumps(wins, indent=2))
    total = sum(w["delta"] for w in wins)
    L(f"DONE winners={len(wins)}  projected +{total:.2f}  ({time.time()-t0:.0f}s)")
    L(f"staged in {STAGE} -> FRESH-GATE via merge_finalize.py before adopt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
