#!/usr/bin/env python3
"""Run the proven task158 fresh dual audit with regolf-local authority paths."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(
    0,
    str(
        ROOT
        / "scripts/golf/loop_8004_42_plus20/agent_task158_current_108"
    ),
)
import fresh_dual as shared  # noqa: E402


shared.HERE = HERE
shared.CANDIDATE = HERE / "sound/task158_exact_regolf.onnx"
shared.DEFAULT_SEEDS = (1_581_141, 1_581_142)


if __name__ == "__main__":
    raise SystemExit(shared.main())
