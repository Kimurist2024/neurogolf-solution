#!/usr/bin/env python3
"""Run the proven ZIP-history half-cost scanner over authority cost 251..500."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import onnxruntime


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def authority_costs() -> dict[int, int]:
    result: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            if 251 <= cost <= 500:
                result[task] = cost
    return result


def main() -> int:
    onnxruntime.set_default_logger_severity(3)
    source_dir = ROOT / "scripts/golf/half_cost_51_100_303"
    common = load("history_scan_307_impl", source_dir / "history_scan.py")
    common.ROOT = ROOT
    common.HERE = HERE
    common.AUTHORITY = ROOT / "submission_base_8011.05.zip"
    common.AUTHORITY_SHA256 = common.sha256(common.AUTHORITY.read_bytes())
    common.authority_costs = authority_costs
    # The user explicitly permits known private-zero lineages at >=95% fresh.
    # Keep them in the evidence scan; final classification remains POLICY95,
    # never guaranteed-safe.
    common.PRIVATE_ZERO_OR_UNSOUND = set()
    sys.modules["history_scan"] = common
    scan = load("zip_half_307_impl", source_dir / "zip_history_scan.py")
    scan.ROOT = ROOT
    scan.OUT = HERE / "zip_half_evidence.json"
    return scan.main()


if __name__ == "__main__":
    raise SystemExit(main())
