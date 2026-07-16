#!/usr/bin/env python3
"""Run conservative ORT optimizer sweeps for high154."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "agent_high216_285_226_136" / "sweep_optimizer.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("high154_optimizer_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.TASKS = (367, 382, 270)
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
