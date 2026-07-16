#!/usr/bin/env python3
"""Re-run the conservative exact Einsum scanner on the 8005.16 payload."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8003_40/agent_exact_resume/build_and_scan.py"
spec = importlib.util.spec_from_file_location("exact_einsum_scan", SOURCE)
assert spec is not None and spec.loader is not None
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)

scanner.HERE = HERE
scanner.ROOT = ROOT
scanner.BASELINE = ROOT / "submission_base_8005.16.zip"
scanner.CANDIDATES = HERE / "candidates"
scanner.EXISTING = {}
scanner.PRIVATE_ZERO_EXCLUDE = set()

scanner.main()
