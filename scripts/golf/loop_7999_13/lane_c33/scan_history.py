#!/usr/bin/env python3
"""Run shared history inventory for task143 and task301."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "lane_c29" / "scan_history.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("c33_history_shared", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError(SHARED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.TARGETS = {143: 212, 301: 240}
    module.main()


if __name__ == "__main__":
    main()
