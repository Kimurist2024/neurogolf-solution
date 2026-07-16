#!/usr/bin/env python3
"""Inventory every historical strict reduction without executing candidates.

The full checker can be dominated by pathological ORT graphs.  This wrapper
reuses the archive scanner but replaces known-case execution with a no-op, so
we can rank candidates first and then audit the best unique artifacts under an
explicit timeout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import strict_history_scan as wrapper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> int:
    source_dir = ROOT / "scripts/golf/half_cost_51_100_303"
    common = wrapper.load("strict_inventory_307_common", source_dir / "history_scan.py")
    common.ROOT = ROOT
    common.HERE = HERE
    common.AUTHORITY = ROOT / "submission_base_8011.05.zip"
    common.AUTHORITY_SHA256 = common.sha256(common.AUTHORITY.read_bytes())
    common.authority_costs = wrapper.authority_costs
    common.PRIVATE_ZERO_OR_UNSOUND = set()
    common.exact_known = lambda model, task: (False, 0, "inventory_only")
    sys.modules["history_scan"] = common
    scan = wrapper.load("strict_inventory_307_impl", source_dir / "strict_history_scan.py")
    scan.ROOT = ROOT
    scan.OUT = HERE / "strict_inventory.json"
    scan.extra_structure = wrapper.allowed_structure
    return scan.main()


if __name__ == "__main__":
    raise SystemExit(main())
