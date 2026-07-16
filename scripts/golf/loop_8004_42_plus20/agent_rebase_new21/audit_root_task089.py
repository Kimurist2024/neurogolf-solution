#!/usr/bin/env python3
"""Independent fail-closed audit of root's 8005.16 task089 exact-shave lead."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import shape_inference

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "base/task089.onnx"
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_exact_noop26/task089.onnx"

sys.path.insert(0, str(HERE))
import audit_baselines as common  # noqa: E402


def compare(base: Any, candidate: Any, rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    result = {
        "total": len(rows),
        "base_right": 0,
        "candidate_right": 0,
        "base_runtime_errors": 0,
        "candidate_runtime_errors": 0,
        "raw_equal": 0,
        "decoded_equal": 0,
        "first_candidate_error": None,
    }
    for index, row in enumerate(rows):
        expected = row["output"] > 0
        base_raw = candidate_raw = None
        try:
            base_raw = base.run(["output"], {"input": row["input"]})[0]
        except Exception:
            result["base_runtime_errors"] += 1
        try:
            candidate_raw = candidate.run(["output"], {"input": row["input"]})[0]
        except Exception as exc:
            result["candidate_runtime_errors"] += 1
            if result["first_candidate_error"] is None:
                result["first_candidate_error"] = {
                    "index": index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if base_raw is not None:
            result["base_right"] += int(
                base_raw.shape == expected.shape and np.array_equal(base_raw > 0, expected)
            )
        if candidate_raw is not None:
            result["candidate_right"] += int(
                candidate_raw.shape == expected.shape
                and np.array_equal(candidate_raw > 0, expected)
            )
        if base_raw is not None and candidate_raw is not None:
            result["raw_equal"] += int(np.array_equal(base_raw, candidate_raw, equal_nan=True))
            result["decoded_equal"] += int(np.array_equal(base_raw > 0, candidate_raw > 0))
    return result


def main() -> int:
    base = onnx.load(BASE)
    candidate = onnx.load(CANDIDATE)
    report: dict[str, Any] = {
        "task": 89,
        "baseline_path": str(BASE.relative_to(ROOT)),
        "candidate_path": str(CANDIDATE.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "baseline_cost": common.cost_of(str(BASE))[2],
        "candidate_cost": common.cost_of(str(CANDIDATE))[2],
        "projected_gain_if_valid": math.log(
            common.cost_of(str(BASE))[2] / common.cost_of(str(CANDIDATE))[2]
        ),
        "structure": {},
        "known": {},
        "fresh_run": False,
        "verdict": "REJECT",
        "reasons": [],
    }
    try:
        onnx.checker.check_model(candidate, full_check=True)
        report["structure"]["checker_full"] = True
    except Exception as exc:
        report["structure"]["checker_full"] = False
        report["structure"]["checker_error"] = f"{type(exc).__name__}: {exc}"
    try:
        shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
        report["structure"]["strict_data_prop"] = True
    except Exception as exc:
        report["structure"]["strict_data_prop"] = False
        report["structure"]["strict_error"] = f"{type(exc).__name__}: {exc}"
    report["structure"]["conv_bias_findings"] = common.check_conv_bias(candidate)
    report["structure"]["standard_domains"] = all(
        item.domain in {"", "ai.onnx"} for item in candidate.opset_import
    ) and all(node.domain in {"", "ai.onnx"} for node in candidate.graph.node)
    report["structure"]["lookup_ops"] = [
        node.op_type
        for node in candidate.graph.node
        if node.op_type in {"TfIdfVectorizer", "Hardmax"}
    ]
    try:
        report["runtime_shapes"] = common.runtime_shape_audit(candidate, 89)
    except Exception as exc:
        report["runtime_shapes"] = {
            "truthful": False,
            "trace_error": f"{type(exc).__name__}: {exc}",
        }
    rows = common.known_rows(89)
    for optimized, label in ((False, "disable_all"), (True, "default")):
        mode_errors: dict[str, str] = {}
        base_session = candidate_session = None
        try:
            base_session = common.session(base, optimized)
        except Exception as exc:
            mode_errors["base_session_error"] = f"{type(exc).__name__}: {exc}"
        try:
            candidate_session = common.session(candidate, optimized)
        except Exception as exc:
            mode_errors["candidate_session_error"] = f"{type(exc).__name__}: {exc}"
        if mode_errors:
            report["known"][label] = mode_errors
            continue
        assert base_session is not None and candidate_session is not None
        report["known"][label] = compare(base_session, candidate_session, rows)
    disabled = report["known"].get("disable_all", {})
    default = report["known"].get("default", {})
    if disabled.get("candidate_runtime_errors", 0) != 0:
        report["reasons"].append("disable_all_runtime_errors_on_all_known_cases")
    if "candidate_session_error" in default:
        report["reasons"].append("default_ORT_session_creation_failed")
    if not report["runtime_shapes"].get("truthful", False):
        report["reasons"].append("runtime_shapes_not_truthful")
    if disabled.get("raw_equal") != len(rows):
        report["reasons"].append("not_raw_equal_to_LB_incumbent")
    report["reasons"].append("fresh_not_run_after_mandatory_known/runtime/shape_gates_failed")
    (HERE / "task089_root_candidate_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
