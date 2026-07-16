#!/usr/bin/env python3
"""Run the exact-rewrite scan on the last two unreported mid-cost targets."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCANNER = HERE.parent / "root_rebase_exact22" / "scan_exact.py"
spec = importlib.util.spec_from_file_location("root_exact_scanner", SCANNER)
assert spec is not None and spec.loader is not None
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)

scanner.HERE = HERE
scanner.CANDIDATES = HERE / "candidates"
scanner.TASKS = (46, 157)

raise SystemExit(scanner.main())
