#!/usr/bin/env python3
"""Inventory only SHA deltas since prior exhaustive campaign scans.

The expensive full-screen lanes already covered almost every target in this
lane.  This script reuses their byte-level inventory implementation, but only
persists metadata for SHAs absent from every older ``rescreen.json``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
TARGETS = (
    239, 222, 37, 226, 297, 14, 234, 92, 397, 264,
    394, 398, 200, 75, 392, 387, 225, 218, 36, 132,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(
    "expand20j_inventory_base",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)
BASE.HERE = HERE
BASE.TARGETS = TARGETS
BASE.BASE_ZIP = AUTHORITY


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    got = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if got != AUTHORITY_SHA256:
        raise SystemExit(f"authority drift: {got}")

    old: dict[int, set[str]] = {task: set() for task in TARGETS}
    old_sources: dict[int, dict[str, list[str]]] = {task: {} for task in TARGETS}
    loaded: list[str] = []
    loop = HERE.parent
    for path in sorted(loop.glob("agent_*/rescreen.json")):
        if HERE in path.parents:
            continue
        try:
            report = json.loads(path.read_text())
        except Exception:
            continue
        used = False
        for row in report.get("rows", []):
            task = int(row.get("task", -1))
            digest = row.get("sha256")
            if task not in old or not digest:
                continue
            old[task].add(digest)
            old_sources[task][digest] = list(row.get("sources", []))
            used = True
        if used:
            loaded.append(str(path.relative_to(ROOT)))

    candidates, inventory = BASE.inventory()
    delta: list[dict] = []
    for task in TARGETS:
        for digest, row in sorted(candidates.get(task, {}).items()):
            if digest in old[task]:
                continue
            delta.append(
                {
                    "task": task,
                    "sha256": digest,
                    "bytes": len(row["data"]),
                    "sources": sorted(set(row["sources"])),
                    "source_kinds": sorted(set(row["source_kinds"])),
                }
            )

    result = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": got,
        "targets": list(TARGETS),
        "prior_rescreens_loaded": loaded,
        "prior_unique_by_task": {str(t): len(old[t]) for t in TARGETS},
        "current_inventory": inventory,
        "delta_unique_by_task": {
            str(t): sum(row["task"] == t for row in delta) for t in TARGETS
        },
        "delta_count": len(delta),
        "delta": delta,
    }
    (HERE / "inventory_delta.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"delta_count": len(delta), "by_task": result["delta_unique_by_task"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
