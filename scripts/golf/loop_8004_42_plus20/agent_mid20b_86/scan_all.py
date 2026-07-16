#!/usr/bin/env python3
"""Exhaustive SHA-deduplicated loose/ZIP rescreen for the mid20b target set.

The implementation reuses the campaign scanner, but derives all baseline costs
from the authoritative 8005.17 archive instead of trusting an older cost map.
It is non-promoting and writes only below this lane directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (
    102, 25, 324, 308, 338, 134, 268, 184, 377, 170,
    239, 222, 48, 234, 264, 200, 387, 132, 388, 228,
)
BASE_ZIP = ROOT / "submission_base_8005.17.zip"
COSTS_PATH = HERE / "baseline_costs_8005_17.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCANNER = load_module(
    "mid20b_scanner",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py",
)
SWEEP = load_module(
    "mid20b_profiler",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_sweep_wave30b/audit_sweep.py",
)


def derive_baseline_costs() -> dict[str, int]:
    costs: dict[str, int] = {}
    profiles: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TARGETS:
            data = archive.read(f"task{task:03d}.onnx")
            profile = SWEEP.profiler_cost(data, task, "baseline_8005_17")
            profiles[str(task)] = profile
            costs[str(task)] = int(profile["cost"])
            print(f"BASE task{task:03d} cost={profile['cost']}", flush=True)
    COSTS_PATH.write_text(
        json.dumps(
            {
                "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
                "costs": costs,
                "profiles": profiles,
            },
            indent=2,
        )
        + "\n"
    )
    return costs


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "candidates").mkdir(exist_ok=True)
    (HERE / "evidence").mkdir(exist_ok=True)
    derive_baseline_costs()
    SCANNER.HERE = HERE
    SCANNER.TARGETS = TARGETS
    SCANNER.BASE_ZIP = BASE_ZIP
    SCANNER.CURRENT_COSTS_JSON = COSTS_PATH
    # The reusable scanner records every result. Its internal label still says
    # fresh500_pass and uses a 95% cut; finalize.py reclassifies at the requested
    # 90% and additionally applies the four-configuration/private-proof gates.
    sys.argv = [sys.argv[0], "--fresh", "500"]
    return int(SCANNER.main())


if __name__ == "__main__":
    raise SystemExit(main())
