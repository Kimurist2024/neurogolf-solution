#!/usr/bin/env python3
"""Re-run loose ONNX history against immutable 8011.05, cost 11..100 only."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import onnxruntime


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_base():
    path = ROOT / "scripts/golf/root_cost50_history_scan_298/scan.py"
    spec = importlib.util.spec_from_file_location("low401_history_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def current_costs() -> dict[int, int]:
    result = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"][4:])
            cost = int(row["cost"])
            if 11 <= cost <= 100:
                result[task] = cost
    return result


def main() -> int:
    base = load_base()
    onnxruntime.set_default_logger_severity(3)
    base.HERE = HERE
    base.ROOT = ROOT
    base.AUTHORITY = ROOT / "submission_base_8011.05.zip"
    base.AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
    base.EVIDENCE = HERE / "current_history_scan.json"
    base.current_costs = current_costs
    original = base.scoring.score_and_verify

    def fail_closed(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception:
            return None

    base.scoring.score_and_verify = fail_closed
    status = base.main()
    payload = json.loads(base.EVIDENCE.read_text(encoding="utf-8"))
    payload["scope"] = "all 148 immutable-authority tasks with cost 11..100"
    payload["authority_lb"] = 8011.05
    payload["half_cost_winners"] = [
        row for row in payload["winners"]
        if int(row["profile"]["cost"]) * 2 <= int(row["authority_cost"])
    ]
    base.EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
