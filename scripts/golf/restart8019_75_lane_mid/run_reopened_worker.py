#!/usr/bin/env python3
"""Reopen the mid-band catalogue tasks, excluding task338 entirely."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent / "reopened"
ROOT = HERE.parents[3]
REOPENED = {48, 168, 170, 178, 185, 192, 222, 354}


def main() -> int:
    source = ROOT / "scripts/golf/restart8018_91_lane_mid/run_worker.py"
    spec = importlib.util.spec_from_file_location("restart8019_mid_reopened_base", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.AUTHORITY = ROOT / "submission_base_8019.75.zip"
    module.AUTHORITY_SHA256 = "e69058edd21e27ab7d32670d714ec5cea6d35632a9d9a620364731297717edb3"
    # The base wrapper expresses eligibility as band minus EXCLUDED.  Exclude
    # every task except this explicit reopened set.  task338 is therefore never
    # scanned and its quarantined cost-334 SHA cannot be reconsidered.
    module.EXCLUDED = set(range(1, 401)) - REOPENED
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
