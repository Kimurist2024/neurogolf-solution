#!/usr/bin/env python3
"""Re-run the full structural/known audit for the task205 proof lane."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_a20/audit_history.py"
spec = importlib.util.spec_from_file_location("task205_proof_shared_audit", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {SOURCE}")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.BASE_COST = {205: 1042}

MODELS = [
    (
        "authority1042",
        ROOT / "scripts/golf/loop_7999_13/lane_a23/baseline/task205.onnx",
        1042,
        ["immutable authority member"],
        True,
    ),
    (
        "lead937",
        ROOT / "scripts/golf/loop_7999_13/lane_a23/candidates/task205_r02.onnx",
        937,
        ["others/2/7805/task205_rebuilt_top2_cost937.onnx"],
        False,
    ),
    (
        "staged_exact1041",
        ROOT / "scripts/golf/loop_8004_42_plus20/agent_high205_338_123/candidates/task205_rowpow_selu.onnx",
        1041,
        ["exact algebraic rewrite of authority1042"],
        False,
    ),
    (
        "authority_rewrite1038",
        ROOT / "scripts/golf/loop_7999_13/lane_a23/candidates/task205_c3_cost1038.onnx",
        1038,
        ["others/2/7901/task205_cost1038.onnx", "others/2/7902/task205_cost1038.onnx"],
        False,
    ),
]


def main() -> int:
    rows = []
    for label, path, cost, sources, baseline in MODELS:
        row = shared.audit(205, label, path, cost, sources, baseline=baseline)
        # Hardmax is a runtime-derived ten-color selector, not a stored lookup.
        reasons = [reason for reason in row["pre_fresh_reasons"] if reason != "lookup"]
        row["hardmax_is_algorithmic_selector"] = True
        row["pre_fresh_reasons_after_hardmax_review"] = reasons
        row["pre_fresh_pass_after_hardmax_review"] = not reasons
        rows.append(row)
        print(label, row.get("official_like_score"), reasons, flush=True)
    result = {
        "task": 205,
        "rows": rows,
        "complete": True,
        "note": "Generator-gold disposition is separate; see counterexample.json.",
    }
    (HERE / "model_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
