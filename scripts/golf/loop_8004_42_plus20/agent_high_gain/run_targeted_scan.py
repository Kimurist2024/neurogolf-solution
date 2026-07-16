#!/usr/bin/env python3
"""Non-promoting archive scan for the high-gain +20 lane."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "scripts/golf/scratch_codex_plus10/wave1_b/scan_candidates.py"

spec = importlib.util.spec_from_file_location("high_gain_scanner", SOURCE)
assert spec is not None and spec.loader is not None
scanner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)

scanner.ROOT = ROOT
scanner.HERE = HERE
scanner.TASKS = (145, 191, 204, 205, 285)
scanner.BASE_ZIP = ROOT / "submission_base_8004.50.zip"
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
