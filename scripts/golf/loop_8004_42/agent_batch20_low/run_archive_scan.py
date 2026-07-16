#!/usr/bin/env python3
"""Run the audited wave scanner for this lane's immutable ten-task list."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "scripts/golf/scratch_codex_plus10/wave1_b/scan_candidates.py"

spec = importlib.util.spec_from_file_location("batch20_scanner", SOURCE)
assert spec is not None and spec.loader is not None
scanner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)

scanner.ROOT = ROOT
scanner.HERE = HERE
scanner.TASKS = (25, 62, 8, 134, 112, 184, 168, 48, 37, 14)
scanner.BASE_ZIP = ROOT / "scripts/golf/loop_8004_42/submission_8004.42_fixed_rebase_meta.zip"
scanner.POOL_ROOTS = (
    ROOT / "others/1/70208",
    ROOT / "others/1/70209",
    ROOT / "others/1/70210",
    ROOT / "others/2/1200",
    ROOT / "others/2/1201",
    ROOT / "others/2/1202",
    ROOT / "others/2/1203",
    ROOT / "others/3",
)
scanner.TREE_ROOTS = (ROOT / "artifacts", ROOT / "scripts/golf")

if __name__ == "__main__":
    raise SystemExit(scanner.main())
