#!/usr/bin/env python3
"""Deep official/shape/known audit of high154 optimizer leads."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "agent_high216_285_226_136" / "audit_optimizer_candidate.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("high154_optimizer_audit_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
