#!/usr/bin/env python3
"""C36 wrapper around the shared dual-ORT fresh generator auditor."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "lane_c21" / "fresh_validate.py"
SPEC = importlib.util.spec_from_file_location("lane_c21_fresh", SOURCE)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)
AUDITOR.HASHES = {12: "0962bcdd"}


if __name__ == "__main__":
    raise SystemExit(AUDITOR.main())
