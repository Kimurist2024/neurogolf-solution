#!/usr/bin/env python3
"""8019.75 wrapper for exact algebraic dedup/static folding."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> int:
    source = ROOT / "scripts/golf/restart8018_91_lane_mid/scan_exact_dedup.py"
    spec = importlib.util.spec_from_file_location("restart8019_mid_exact_base", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.AUTHORITY = ROOT / "submission_base_8019.75.zip"
    module.AUTHORITY_SHA256 = "e69058edd21e27ab7d32670d714ec5cea6d35632a9d9a620364731297717edb3"
    # Exact algebra relative to a verified white authority is safe to inspect
    # on every task; admission still requires all mandatory runtime gates.
    module.EXCLUDED = set()
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
