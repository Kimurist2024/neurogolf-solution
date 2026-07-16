#!/usr/bin/env python3
"""Audit the immutable 8005.16 members for the high48 target lane."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "agent_new_low45" / "audit_latest.py"
SPEC = importlib.util.spec_from_file_location("high48_shared_latest", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(SOURCE)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)

shared.HERE = HERE
shared.ROOT = HERE.parents[3]
shared.BASE = shared.ROOT / "submission_base_8005.16.zip"
shared.PREVIOUS = shared.ROOT / "submission_base_8004.50.zip"
shared.TARGETS = (8, 275, 134, 112, 168, 109, 160, 170)


if __name__ == "__main__":
    shared.main()
