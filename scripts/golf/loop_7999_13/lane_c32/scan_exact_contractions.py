#!/usr/bin/env python3
"""Reuse the exact constant-pair contraction analyzer for C32 baselines."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "lane_c31" / "find_exact_contractions.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("c32_contraction_shared", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError(SHARED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {
        path.stem: module.inspect(path)
        for path in sorted((HERE / "baseline").glob("*.onnx"))
    }
    (HERE / "exact_contractions.json").write_text(json.dumps(result, indent=2) + "\n")
    for task, row in result.items():
        reducing = [item for item in row["pair_contractions"] if item["delta"] < 0]
        print(task, "pairs", len(row["pair_contractions"]), "reducing", reducing)


if __name__ == "__main__":
    main()
