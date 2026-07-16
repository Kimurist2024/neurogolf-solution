#!/usr/bin/env python3
"""Run the exhaustive loose/ZIP scanner for the mid20_84 target set."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py"
SPEC = importlib.util.spec_from_file_location("mid20_84_scanner", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load exhaustive scanner")
SCANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCANNER)

SCANNER.HERE = HERE
SCANNER.TARGETS = (
    374, 250, 62, 8, 275, 112, 168, 109, 160, 99,
    279, 345, 245, 37, 297, 14, 92, 397, 394, 398,
)
SCANNER.BASE_ZIP = ROOT / "submission_base_8005.17.zip"
# 8005.17 differs from 8004.50 only at task226, outside this target set.
SCANNER.CURRENT_COSTS_JSON = ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json"


def main() -> int:
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    (HERE / "evidence").mkdir(parents=True, exist_ok=True)
    # The reusable scanner's initial fresh gate is 95%.  We retain all rows in
    # rescreen.json and reclassify at the requested 90% in confirm_policy90.py.
    sys.argv = [sys.argv[0], "--fresh", "500"]
    return int(SCANNER.main())


if __name__ == "__main__":
    raise SystemExit(main())
