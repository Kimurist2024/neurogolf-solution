#!/usr/bin/env python3
"""Exhaustive authority-cost scan for the fifth 20-task target set."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_near95_wave2/rescreen_all.py"
SPEC = importlib.util.spec_from_file_location("mid20e89_authority_scanner", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load exhaustive scanner")
SCANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCANNER)

SCANNER.HERE = HERE
SCANNER.TARGETS = (
    68, 175, 400, 30, 224, 281, 240, 183, 376, 59,
    358, 20, 190, 302, 195, 300, 383, 193, 304, 384,
)
SCANNER.BASE_ZIP = ROOT / "submission_base_8006.61.zip"
SCANNER.CURRENT_COSTS_JSON = HERE / "authority_costs.json"

PRIVATE_OR_UNSOUND = {302}
_inventory = SCANNER.inventory


def inventory_with_catalog():
    candidates, report = _inventory()
    for task in PRIVATE_OR_UNSOUND:
        for item in candidates.get(task, {}).values():
            item["sources"].append(f"private_zero_catalog_task{task:03d}")
    report["forced_private_zero_catalog"] = sorted(PRIVATE_OR_UNSOUND)
    return candidates, report


SCANNER.inventory = inventory_with_catalog

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

# Four complete known runs: two isolated session constructions for each ORT
# mode. The reusable scanner treats every returned mode as a mandatory pass.
_known_dual = SCANNER.known_dual


def known_quad(task, data):
    first = _known_dual(task, data)
    second = _known_dual(task, data)
    return {
        "disable_all_run1": first["disable_all"],
        "default_run1": first["default"],
        "disable_all_run2": second["disable_all"],
        "default_run2": second["default"],
    }


SCANNER.known_dual = known_quad


def main() -> int:
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    (HERE / "evidence").mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--fresh", "500"]
    return int(SCANNER.main())


if __name__ == "__main__":
    raise SystemExit(main())
