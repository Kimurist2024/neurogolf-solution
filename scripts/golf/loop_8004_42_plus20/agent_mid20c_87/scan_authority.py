#!/usr/bin/env python3
"""Exhaustive loose/ZIP SHA scan for the third 20-task expansion set."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py"
SPEC = importlib.util.spec_from_file_location("mid20c87_authority_scanner", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load exhaustive scanner")
SCANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCANNER)

SCANNER.HERE = HERE
SCANNER.TARGETS = (
    51, 64, 29, 178, 123, 91, 124, 148, 199, 341,
    357, 137, 355, 169, 316, 212, 301, 174, 153, 325,
)
SCANNER.BASE_ZIP = ROOT / "submission_base_8005.17.zip"
SCANNER.CURRENT_COSTS_JSON = HERE / "authority_costs.json"

# These tasks are in the documented private-zero/unsound catalog. A copied SHA
# in an ordinary ZIP does not erase the lineage, and no complete true-rule
# proof accompanies any harvested candidate.
PRIVATE_OR_UNSOUND = {178, 169, 174, 325}
_inventory = SCANNER.inventory


def inventory_with_catalog():
    candidates, report = _inventory()
    for task in PRIVATE_OR_UNSOUND:
        for item in candidates.get(task, {}).values():
            item["sources"].append(f"private_zero_catalog_task{task:03d}")
    report["forced_private_zero_catalog"] = sorted(PRIVATE_OR_UNSOUND)
    return candidates, report


SCANNER.inventory = inventory_with_catalog

# onnx.checker currently accepts ORT's negative-Conv-pad crop extension even
# though Conv's schema requires non-negative pads. Fail closed explicitly.
_strict_extra = SCANNER.strict_extra


def strict_extra_schema(data, sources):
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


SCANNER.strict_extra = strict_extra_schema


def main() -> int:
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    (HERE / "evidence").mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--fresh", "500"]
    return int(SCANNER.main())


if __name__ == "__main__":
    raise SystemExit(main())
