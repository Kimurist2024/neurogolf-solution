#!/usr/bin/env python3
"""Run semantics-preserving ONNX optimizer passes on safe cost-250..299 tasks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent / "optimizer_band"
ROOT = HERE.parents[3]
TASKS = (51, 64, 75, 123, 124, 132, 148, 159, 199, 228, 398)


def main() -> int:
    source = ROOT / "scripts/golf/restart8023_08_lane_low/scan_exact_optimizers.py"
    spec = importlib.util.spec_from_file_location("restart8023_mid_optimizer_base", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.TASKS = TASKS
    module.BASE = HERE / "base"
    module.CANDIDATES = HERE / "candidates"
    base_profile = module.profile

    def guarded_profile(model, task, label):
        result = base_profile(model, task, label)
        if min(result["memory"], result["params"], result["cost"]) < 0:
            raise RuntimeError("official scorer rejected model (negative sentinel profile)")
        return result

    module.profile = guarded_profile
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
