#!/usr/bin/env python3
"""Static/full-check audit of task196 controls and sound fallback."""

from __future__ import annotations

import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODELS = {
    "authority": HERE / "baseline_task196.onnx",
    "historical_968": ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task196_r07_static296.onnx",
    "truthful_authority": HERE / "truthful_authority.onnx",
    "truthful_historical_968": HERE / "truthful_historical_968.onnx",
    "sound_bitset_u16": ROOT / "scripts/golf/scratch_codex/task196/agent_bitset/candidate_bitset_u16.onnx",
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def shape_issues(model: onnx.ModelProto) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        current = []
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            current.append("missing_shape")
        else:
            for dim in tensor.shape.dim:
                if dim.HasField("dim_param"):
                    current.append(f"dim_param:{dim.dim_param}")
                elif not dim.HasField("dim_value") or dim.dim_value <= 0:
                    current.append("nonpositive_or_missing_dim")
        if current:
            issues[value.name] = current
    return issues


def main() -> None:
    rows = {}
    for label, path in MODELS.items():
        model = onnx.load(path)
        initializers = {init.name: init for init in model.graph.initializer}
        checker = strict = True
        checker_error = strict_error = None
        try:
            onnx.checker.check_model(model, full_check=True)
        except Exception as exc:
            checker = False
            checker_error = repr(exc)
        try:
            inferred = onnx.shape_inference.infer_shapes(
                model, strict_mode=True, data_prop=True
            )
        except Exception as exc:
            strict = False
            strict_error = repr(exc)
            inferred = model
        nested = []
        bias_findings = []
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                    nested.append({"node": node.name, "attribute": attr.name})
            if node.op_type == "Conv" and len(node.input) >= 3 and node.input[2]:
                weight = initializers.get(node.input[1])
                bias = initializers.get(node.input[2])
                if weight is not None and bias is not None and list(bias.dims) != [weight.dims[0]]:
                    bias_findings.append({"node": node.name, "weight": list(weight.dims), "bias": list(bias.dims)})
            if node.op_type == "QLinearConv" and len(node.input) >= 9 and node.input[8]:
                weight = initializers.get(node.input[3])
                bias = initializers.get(node.input[8])
                if weight is not None and bias is not None and list(bias.dims) != [weight.dims[0]]:
                    bias_findings.append({"node": node.name, "weight": list(weight.dims), "bias": list(bias.dims)})
        rows[label] = {
            "path": str(path.relative_to(ROOT)),
            "full_checker": checker,
            "checker_error": checker_error,
            "strict_shape_data_prop": strict,
            "strict_error": strict_error,
            "shape_issues": shape_issues(inferred),
            "banned_ops": sorted(
                {node.op_type for node in model.graph.node if node.op_type.upper() in BANNED}
            ),
            "sequence_ops": sorted(
                {node.op_type for node in model.graph.node if "Sequence" in node.op_type}
            ),
            "nested_graph_attributes": nested,
            "conv_bias_findings": bias_findings,
            "functions": len(model.functions),
            "sparse_initializers": len(model.graph.sparse_initializer),
            "nonstandard_domains": sorted(
                {opset.domain for opset in model.opset_import if opset.domain not in {"", "ai.onnx"}}
            ),
        }
    (HERE / "structural_audit.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
