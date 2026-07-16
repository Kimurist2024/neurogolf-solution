#!/usr/bin/env python3
"""C22 wrapper around the dual-ORT fresh generator auditor."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "lane_c21" / "fresh_validate.py"
SPEC = importlib.util.spec_from_file_location("lane_c21_fresh", SOURCE)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)
AUDITOR.HASHES = {9: "06df4c85", 76: "36d67576"}


if __name__ == "__main__":
    raise SystemExit(AUDITOR.main())
