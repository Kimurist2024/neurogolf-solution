#!/usr/bin/env python3
"""Run the 8005.16 mechanical exact scan on twelve new low-cost targets."""

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
scanner.TASKS = (161, 189, 304, 384, 193, 175, 30, 281, 376, 20, 302, 195)

raise SystemExit(scanner.main())
