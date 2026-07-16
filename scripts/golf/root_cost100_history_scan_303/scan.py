#!/usr/bin/env python3
"""Run the proven history scan against the current 8011.05 cost<=100 scope."""

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
    spec = importlib.util.spec_from_file_location("history_scan_303_base", path)
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
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            if cost <= 100 and float(row["score"]) < 25.0:
                result[task] = cost
    return result


def main() -> int:
    base = load_base()
    onnxruntime.set_default_logger_severity(3)
    authority = ROOT / "submission_base_8011.05.zip"
    base.HERE = HERE
    base.ROOT = ROOT
    base.AUTHORITY = authority
    base.AUTHORITY_SHA256 = base.sha256(authority.read_bytes())
    base.EVIDENCE = HERE / "evidence.json"
    base.current_costs = current_costs
    original_score = base.scoring.score_and_verify

    def fail_closed_score(*args, **kwargs):
        try:
            return original_score(*args, **kwargs)
        except Exception:
            return None

    base.scoring.score_and_verify = fail_closed_score
    status = base.main()
    payload = json.loads(base.EVIDENCE.read_text())
    payload["scope"] = "all 165 non-score25 authority tasks with cost<=100"
    payload["authority_lb"] = 8011.05
    payload["half_cost_winners"] = [
        row for row in payload["winners"]
        if int(row["profile"]["cost"]) * 2 <= int(row["authority_cost"])
    ]
    base.EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
