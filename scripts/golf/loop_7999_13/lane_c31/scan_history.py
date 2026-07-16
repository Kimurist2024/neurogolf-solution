#!/usr/bin/env python3
"""Run the shared loose-history inventory for task199 and task212."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "lane_c29" / "scan_history.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("c31_history_shared", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError(SHARED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.TARGETS = {199: 261, 212: 240}
    module.main()


if __name__ == "__main__":
    main()
