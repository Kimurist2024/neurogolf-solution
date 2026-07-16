#!/usr/bin/env python3
"""Run the exhaustive 8005.17 loose/ZIP scanner for mid20d_88."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (
    55, 31, 86, 88, 42, 143, 247, 79, 65, 344,
    115, 163, 206, 114, 273, 161, 71, 105, 259, 189,
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
    "mid20d_scan_base",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/scan_all.py",
)
BASE.HERE = HERE
BASE.TARGETS = TARGETS
BASE.BASE_ZIP = ROOT / "submission_base_8005.17.zip"
BASE.COSTS_PATH = HERE / "baseline_costs_8005_17.json"


if __name__ == "__main__":
    raise SystemExit(BASE.main())
