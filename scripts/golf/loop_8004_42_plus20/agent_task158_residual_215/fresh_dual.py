#!/usr/bin/env python3
"""Fresh dual-mode raw-equivalence audit for the exact role-bit rewrite."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(
    0,
    str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_task158_current_108"),
)
import fresh_dual as shared  # noqa: E402


shared.HERE = HERE
shared.CANDIDATE = HERE / "candidates/task158_exact_anchor_role_bits.onnx"
shared.TRUSTED = ROOT / "others/71407/task158.onnx"
shared.DEFAULT_SEEDS = (1_582_151, 1_582_152)


if __name__ == "__main__":
    raise SystemExit(shared.main())
