#!/usr/bin/env python3
"""Run the established fail-closed task192 audit on threshold k=31."""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_sound192_344_93/audit_task192_exact_poly.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("root188_task192_audit", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.CANDIDATE = HERE / "candidates/task192_hardsigmoid_k31.onnx"
    audit.HERE = HERE
    return int(audit.main())


if __name__ == "__main__":
    raise SystemExit(main())
