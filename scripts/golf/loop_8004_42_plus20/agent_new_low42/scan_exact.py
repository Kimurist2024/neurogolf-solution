#!/usr/bin/env python3
"""Run the shared narrow exact-rewrite scan for low42."""

from pathlib import Path
import importlib.util


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "root_rebase_exact22" / "scan_exact.py"
SPEC = importlib.util.spec_from_file_location("root_rebase_exact22", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MODULE.HERE = HERE
MODULE.CANDIDATES = HERE / "candidates"
MODULE.TASKS = (339, 126, 21, 171, 346, 227, 318, 332)


if __name__ == "__main__":
    raise SystemExit(MODULE.main())
