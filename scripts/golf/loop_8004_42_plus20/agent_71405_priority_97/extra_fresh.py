#!/usr/bin/env python3
"""Reopen nonfinite diagnostic candidates for fresh ranking."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCANNER = load_module(
    "priority97_extra_fresh_base",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)


def run(task: int, path: Path, count: int, seeds: tuple[int, ...]) -> dict:
    data = path.read_bytes()
    import hashlib
    digest = hashlib.sha256(data).hexdigest()
    candidate = [{"sha256": digest, "data": data, "sources": [str(path.relative_to(ROOT))]}]
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "count": count,
        "runs": [SCANNER.fresh_dual(task, candidate, count, seed) for seed in seeds],
    }


def main() -> int:
    rows = [
        run(13, ROOT / "others/71405/task013_cost357.onnx", 500, (202607149703, 202607149704)),
        run(46, ROOT / "others/71405/task046_reimproved.onnx", 5000, (202607149705,)),
    ]
    (HERE / "audit" / "extra_fresh.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
