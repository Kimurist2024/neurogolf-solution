#!/usr/bin/env python3
"""Re-run the 20-task inventory against costs profiled from the exact ZIP."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
OUT = HERE / "authority_rescan"
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py"
SPEC = importlib.util.spec_from_file_location("mid20_84_authority_scanner", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load exhaustive scanner")
SCANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCANNER)

SCANNER.HERE = OUT
SCANNER.TARGETS = (
    374, 250, 62, 8, 275, 112, 168, 109, 160, 99,
    279, 345, 245, 37, 297, 14, 92, 397, 394, 398,
)
SCANNER.BASE_ZIP = ROOT / "submission_base_8005.17.zip"
SCANNER.CURRENT_COSTS_JSON = HERE / "authority_costs.json"

# Authority task112 and task168 have documented private-zero/unsound history.
# SHA copies in ordinary archives must not erase that lineage.
_inventory = SCANNER.inventory


def inventory_with_catalog():
    candidates, report = _inventory()
    for task in (112, 168):
        for item in candidates.get(task, {}).values():
            item["sources"].append(f"private_zero_catalog_task{task:03d}")
    report["forced_private_zero_catalog"] = [112, 168]
    return candidates, report


SCANNER.inventory = inventory_with_catalog

# The generic reusable scanner predates the explicit non-negative Conv pads
# gate.  Enforce the ONNX schema contract here even when onnx.checker accepts a
# runtime extension.
_strict_extra = SCANNER.strict_extra


def strict_extra_with_nonnegative_pads(data, sources):
    passed, reasons, detail = _strict_extra(data, sources)
    model = onnx.load_model_from_string(data)
    findings = []
    for node in model.graph.node:
        if node.op_type not in {"Conv", "ConvTranspose"}:
            continue
        for attr in node.attribute:
            if attr.name == "pads" and any(value < 0 for value in attr.ints):
                findings.append({"output": node.output[0], "pads": list(attr.ints)})
    detail["negative_conv_pads"] = findings
    if findings:
        reasons = sorted(set([*reasons, "negative_conv_pads"]))
        passed = False
    return passed, reasons, detail


SCANNER.strict_extra = strict_extra_with_nonnegative_pads


def main() -> int:
    (OUT / "candidates").mkdir(parents=True, exist_ok=True)
    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--fresh", "500"]
    return int(SCANNER.main())


if __name__ == "__main__":
    raise SystemExit(main())
