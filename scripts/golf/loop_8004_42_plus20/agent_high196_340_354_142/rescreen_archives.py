#!/usr/bin/env python3
"""Revalidate all historically retained lower-floor payloads for high142."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE.parent / "agent_high216_285_226_136" / "rescreen_archives.py"
INVENTORY = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json"


def retained_paths() -> dict[int, list[Path]]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return {
        task: [ROOT / row["path"] for row in inventory["retained"].get(str(task), [])]
        for task in (196, 340, 354)
    }


def main() -> int:
    spec = importlib.util.spec_from_file_location("high142_archive_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.candidates = retained_paths
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
