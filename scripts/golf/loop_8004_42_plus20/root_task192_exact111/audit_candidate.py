#!/usr/bin/env python3
"""Run the fail-closed task192 audit against the immutable 8008.14 base."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_sound192_344_93/audit_task192_exact_poly.py"


def load_source():
    spec = importlib.util.spec_from_file_location("root111_sound93_audit", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    audit = load_source()
    audit.HERE = HERE
    audit.ROOT = ROOT
    audit.CANDIDATE = HERE / "task192_selected_masks.onnx"
    audit.AUTHORITY = ROOT / "submission_base_8008.14.zip"
    audit.AUTHORITY_SHA256 = "50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6"
    audit.BASELINE_MEMBER_SHA256 = "e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c"
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
