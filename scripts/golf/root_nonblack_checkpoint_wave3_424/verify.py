#!/usr/bin/env python3
"""Run the pinned-authority verifier against the wave3 checkpoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/golf/root_nonblack_checkpoint_wave2_415/verify.py"
spec = importlib.util.spec_from_file_location("wave2_verify_for_wave3", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.CHECKPOINT = ROOT / "others/71407/nonblack_policy90_8012_15_wave3"
raise SystemExit(module.main())
