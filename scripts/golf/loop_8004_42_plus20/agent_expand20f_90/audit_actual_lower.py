#!/usr/bin/env python3
"""Run known×4 and direct runtime-shape audits for expand20f_90."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (
    75, 392, 225, 218, 159, 185, 263, 370, 182, 330,
    361, 157, 280, 382, 201, 251, 12, 107, 131, 364,
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
    "expand20f_audit_base",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/audit_actual_lower.py",
)
BASE.HERE = HERE
BASE.TARGETS = TARGETS
BASE.BASE_ZIP = ROOT / "submission_base_8006.61.zip"
BASE.COSTS_PATH = HERE / "baseline_costs_8006_61.json"


if __name__ == "__main__":
    raise SystemExit(BASE.main())
