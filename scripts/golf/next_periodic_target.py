#!/usr/bin/env python3
"""Pick the next Kimi target for the periodic adder (one task per call).

Selection zone is the *completable headroom* band: tasks whose LIVE current
cost (from artifacts/handcrafted/, falling back to the best submission zip) is
in [LO, HI]. Live cost is used — NOT the stale docs/golf/real_incumbent.json —
so an already-optimized task (e.g. task297 at 1912 despite a stale 9171) is
skipped instead of wasting a worker.

Ordering: cost DESCENDING (lowest score first = most headroom / biggest single
win potential), but the band caps out below the uncompletable >HI monsters that
returned zero promotions in the 84k-119k wave.

Blocked: docs/golf/kimi_exclude.json (solo/factory-owned) and
docs/golf/periodic_launched.json (already launched by this daemon). The
attempted list is NOT blocking — re-attempt is allowed (a fresh worker may
crack a task a prior one could not), and the live-cost check prevents wasting
effort on tasks that no longer have headroom.

Prints one `task:hash:cost` triple (live cost), or nothing when the band is dry.

Env: KIMI_LO (default 5000), KIMI_HI (default 30000).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx

REPO = Path(__file__).resolve().parents[2]
FS = REPO / "docs" / "golf"
HAND = REPO / "artifacts" / "handcrafted"
BEST_ZIP = REPO / "artifacts" / "_BEST_6776.26.zip"

sys.path.insert(0, str(REPO / "scripts"))
from lib import scoring  # noqa: E402


def _load_json(path: Path, default):
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return default


def live_cost(task: int, best: zipfile.ZipFile | None) -> int | None:
    """Cheapest valid cost across handcrafted and the best-submission zip."""
    costs: list[int] = []
    hp = HAND / f"task{task:03d}.onnx"
    sources: list[onnx.ModelProto] = []
    if hp.is_file():
        try:
            sources.append(onnx.load(str(hp)))
        except Exception:  # noqa: BLE001
            pass
    if best is not None:
        try:
            sources.append(onnx.load_model_from_string(best.read(f"task{task:03d}.onnx")))
        except Exception:  # noqa: BLE001
            pass
    for model in sources:
        with tempfile.TemporaryDirectory() as wd:
            scored = scoring.score_and_verify(
                model, task, wd, label="live", require_correct=False
            )
        if scored:
            costs.append(scored["cost"])
    return min(costs) if costs else None


def main() -> int:
    lo = float(os.environ.get("KIMI_LO", "5000"))
    hi = float(os.environ.get("KIMI_HI", "130000"))
    running_window = float(os.environ.get("KIMI_RUNNING_SEC", "300"))

    inc = _load_json(FS / "real_incumbent.json", {})
    hashes = _load_json(FS / "task_hash_map.json", {})
    launched = set(_load_json(FS / "periodic_launched.json", []))

    # A task is "currently running" if its worker log was written very recently.
    logs_dir = REPO / "artifacts" / "kimi_logs"
    now = max((p.stat().st_mtime for p in logs_dir.glob("task*.log")), default=0.0)
    running: set[int] = set()
    for p in logs_dir.glob("task*.log"):
        try:
            n = int(p.stem.replace("task", ""))
        except ValueError:
            continue
        if now - p.stat().st_mtime <= running_window:
            running.add(n)

    blocked = launched | running  # exclude/attempted are NOT blocking; the live
    # cost gate below auto-skips tasks already optimized below `lo`.

    best = zipfile.ZipFile(BEST_ZIP) if BEST_ZIP.is_file() else None

    # Stale cost descending = biggest-headroom (lowest-score) first. Confirm with
    # live cost so already-optimized tasks (e.g. task297) are skipped.
    ranked = sorted(((int(t), c) for t, c in inc.items()), key=lambda kv: -kv[1])
    for t, stale in ranked:
        if t in blocked:
            continue
        if stale < lo * 0.6:  # cheap prefilter; nothing below the band remains
            break
        lc = live_cost(t, best)
        if lc is None:
            continue
        if not (lo <= lc <= hi):  # live headroom gate
            continue
        h = hashes.get(f"{t:03d}") or hashes.get(str(t))
        if not h:
            continue
        print(f"{t}:{h}:{lc}")
        return 0
    return 0  # nothing with live headroom available -> empty output


if __name__ == "__main__":
    raise SystemExit(main())
