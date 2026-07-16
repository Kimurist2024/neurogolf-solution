#!/usr/bin/env python3
"""Audit low42 targets against the immutable 8005.16 baseline."""

from pathlib import Path
import importlib.util


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "agent_new_low35" / "audit_latest.py"
SPEC = importlib.util.spec_from_file_location("low35_audit_latest", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MODULE.HERE = HERE
MODULE.TARGETS = (339, 126, 21, 171, 346, 227, 318, 332)


if __name__ == "__main__":
    MODULE.main()
