#!/usr/bin/env python3
"""Strict structure, cost, known-case, and runtime-shape audit for task343."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDITOR = HERE.parent / "lane_c11" / "audit_candidates.py"
MODELS = {
    "baseline": HERE / "baseline" / "task343.onnx",
    "candidate": HERE / "candidate" / "task343.onnx",
}


def load_auditor():
    spec = importlib.util.spec_from_file_location("c39_c11_auditor", AUDITOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inferred_shapes_static(path: Path) -> bool:
    model = onnx.shape_inference.infer_shapes(
        onnx.load(path), strict_mode=True, data_prop=True
    )
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            return False
        if any(
            not dim.HasField("dim_value") or dim.dim_value <= 0
            for dim in tensor.shape.dim
        ):
            return False
    return True


def safe_gate(row: dict[str, Any], path: Path) -> dict[str, Any]:
    trace = row["runtime_shape_trace"]
    score = row["official_like_score"]
    ops = row["op_histogram"]
    lookup_ops = {
        name: ops.get(name, 0)
        for name in ("TfIdfVectorizer", "Hardmax", "GatherND", "TopK", "ScatterND")
        if ops.get(name, 0)
    }
    checks = {
        "full_checker": row["full_check"],
        "strict_inference": row["strict_shape_data_prop"],
        "all_inferred_shapes_static_positive": inferred_shapes_static(path),
        "runtime_declared_shape_mismatches_zero": not trace["declared_actual_mismatches"],
        "truthful_runtime_bytes_equal_scored_memory": trace[
            "single_example_intermediate_bytes"
        ]
        == score["memory"],
        "standard_domains": not row["nonstandard_domains"],
        "no_banned_ops": not row["banned_ops"],
        "no_nested_graphs": row["nested_graph_attributes"] == 0,
        "no_functions": row["function_count"] == 0,
        "no_sparse_initializers": row["sparse_initializer_count"] == 0,
        "conv_bias_safe": not row["conv_bias_findings"],
        "no_lookup_red_flags": not lookup_ops,
        "no_giant_einsum": row["lookup_red_flags"]["giant_einsum_nodes"] == 0,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "lookup_ops": lookup_ops,
        "gather_explanation": (
            "The sole Gather applies the Mod-generated periodic coordinate vector to the input. "
            "It is an algorithmic axis remap, not a stored output lookup table."
        ),
        "dynamic_conv_explanation": (
            "Each Conv reuses the input as its weight to compute a scalar self-correlation feature; "
            "there is no bias input and no out-of-bounds bias dependency."
        ),
        "shape_cloak": bool(trace["declared_actual_mismatches"])
        or trace["single_example_intermediate_bytes"] != score["memory"],
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    auditor = load_auditor()
    output: dict[str, Any] = {"task": 343, "models": {}}
    for label, path in MODELS.items():
        row = auditor.audit(f"lane_c39_{label}_task343", 343, path)
        row["safe_structure"] = safe_gate(row, path)
        output["models"][label] = row
    base = output["models"]["baseline"]
    cand = output["models"]["candidate"]
    output["comparison"] = {
        "baseline_cost": base["official_like_score"]["cost"],
        "candidate_cost": cand["official_like_score"]["cost"],
        "cost_reduction": base["official_like_score"]["cost"]
        - cand["official_like_score"]["cost"],
        "projected_score_gain": cand["official_like_score"]["score"]
        - base["official_like_score"]["score"],
        "both_safe_structure": base["safe_structure"]["pass"]
        and cand["safe_structure"]["pass"],
    }
    (HERE / "structural_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
