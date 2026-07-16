#!/usr/bin/env python3
"""Fail-closed history rebase for authority tasks whose cost is 251..500.

Only candidates whose cheap structural lower bound can meet the requested
half-cost target are admitted to expensive known-corpus profiling.
"""

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
    spec = importlib.util.spec_from_file_location("history_half_307_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def current_costs() -> dict[int, int]:
    result: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            if 251 <= cost <= 500:
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
    base.EVIDENCE = HERE / "history_half_evidence.json"
    base.current_costs = current_costs

    # The imported scanner normally asks only for strict improvement.  Tighten
    # its lower-bound filter to half the incumbent while retaining its final
    # official-like runtime profile and known-corpus exactness check.
    original_declared_lower_bound = base.declared_lower_bound

    def half_lower_bound(model):
        return 2 * original_declared_lower_bound(model)

    base.declared_lower_bound = half_lower_bound

    original_score = base.scoring.score_and_verify

    def fail_closed_score(*args, **kwargs):
        try:
            return original_score(*args, **kwargs)
        except Exception:
            return None

    base.scoring.score_and_verify = fail_closed_score
    status = base.main()
    payload = json.loads(base.EVIDENCE.read_text(encoding="utf-8"))
    payload["scope"] = "all 61 authority tasks with cost 251..500"
    payload["authority_lb"] = 8011.05
    payload["note"] = (
        "declared_lower_bound stored by the imported scanner is doubled; "
        "original structural lower bound is declared_lower_bound//2"
    )
    payload["half_cost_winners"] = [
        row
        for row in payload["winners"]
        if row.get("profile") is not None
        and int(row["profile"]["cost"]) * 2 <= int(row["authority_cost"])
    ]
    base.EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
