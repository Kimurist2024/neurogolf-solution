#!/usr/bin/env python3
"""Broaden history rebase to every strict reduction for cost 251..500."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
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


def allowed_structure(model: onnx.ModelProto) -> list[str]:
    """Reject correctness/cost hazards, but allow lookup/private-zero formulas."""
    reasons = []
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        reasons.append("noncanonical_io_count")
    else:
        def dims(value):
            return [int(d.dim_value) if d.HasField("dim_value") else None
                    for d in value.type.tensor_type.shape.dim]
        if dims(model.graph.input[0]) != [1, 10, 30, 30]:
            reasons.append("noncanonical_input_shape")
        if dims(model.graph.output[0]) != [1, 10, 30, 30]:
            reasons.append("noncanonical_output_shape")
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    for tensor in model.graph.initializer:
        try:
            if not np.all(np.isfinite(onnx.numpy_helper.to_array(tensor))):
                reasons.append("nonfinite_initializer")
        except Exception:
            reasons.append("unreadable_initializer")
    for node in model.graph.node:
        if node.op_type in {"Conv", "ConvTranspose"} and len(node.input) >= 3 and node.input[2]:
            weight = initializers.get(node.input[1])
            bias = initializers.get(node.input[2])
            if weight is not None and bias is not None and weight.dims and bias.dims:
                channels = int(weight.dims[0] if node.op_type == "Conv" else weight.dims[1])
                if int(bias.dims[0]) != channels:
                    reasons.append("conv_bias_ub")
    return sorted(set(reasons))


def main() -> int:
    onnxruntime.set_default_logger_severity(3)
    source_dir = ROOT / "scripts/golf/half_cost_51_100_303"
    common = load("strict_history_307_common", source_dir / "history_scan.py")
    common.ROOT = ROOT
    common.HERE = HERE
    common.AUTHORITY = ROOT / "submission_base_8011.05.zip"
    common.AUTHORITY_SHA256 = common.sha256(common.AUTHORITY.read_bytes())
    common.authority_costs = authority_costs
    common.PRIVATE_ZERO_OR_UNSOUND = set()
    sys.modules["history_scan"] = common
    scan = load("strict_history_307_impl", source_dir / "strict_history_scan.py")
    scan.ROOT = ROOT
    scan.OUT = HERE / "strict_history_evidence.json"
    scan.extra_structure = allowed_structure
    return scan.main()


if __name__ == "__main__":
    raise SystemExit(main())
