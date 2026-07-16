#!/usr/bin/env python3
"""Run the deterministic exact-graph audit for the B11 task set."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_b10"))
import audit_exact_graphs as audit  # noqa: E402


audit.LANE = HERE
audit.TASKS = (264, 281, 300, 358, 376, 387, 392)


if __name__ == "__main__":
    audit.main()
