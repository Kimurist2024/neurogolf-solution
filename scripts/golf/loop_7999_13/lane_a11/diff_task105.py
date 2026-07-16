#!/usr/bin/env python3
"""Record protobuf-level and executable-graph differences for task105."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "baseline" / "task105.onnx"
CAND = HERE / "candidate_task105_static198.onnx"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    output: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            output.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            output.append(dim.dim_param)
        else:
            output.append("")
    return output


def initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def node_rows(model: onnx.ModelProto) -> list[dict[str, object]]:
    return [
        {
            "op_type": node.op_type,
            "domain": node.domain,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {
                attr.name: repr(helper.get_attribute_value(attr)) for attr in node.attribute
            },
        }
        for node in model.graph.node
    ]


def cleared_vi_bytes(model: onnx.ModelProto) -> bytes:
    clone = copy.deepcopy(model)
    del clone.graph.value_info[:]
    return clone.SerializeToString()


def main() -> None:
    base = onnx.load(BASE)
    cand = onnx.load(CAND)
    base_init = initializer_map(base)
    cand_init = initializer_map(cand)
    init_diffs = []
    for name in sorted(set(base_init) | set(cand_init)):
        left = base_init.get(name)
        right = cand_init.get(name)
        if left is None or right is None or left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            init_diffs.append(
                {
                    "name": name,
                    "baseline": None if left is None else {"shape": list(left.shape), "dtype": str(left.dtype), "elements": int(left.size)},
                    "candidate": None if right is None else {"shape": list(right.shape), "dtype": str(right.dtype), "elements": int(right.size)},
                    "values_equal": bool(left is not None and right is not None and np.array_equal(left, right)),
                }
            )
    base_nodes = node_rows(base)
    cand_nodes = node_rows(cand)
    node_diffs = []
    for index in range(max(len(base_nodes), len(cand_nodes))):
        left = base_nodes[index] if index < len(base_nodes) else None
        right = cand_nodes[index] if index < len(cand_nodes) else None
        if left != right:
            node_diffs.append({"index": index, "baseline": left, "candidate": right})
    base_vi = {item.name: dims(item) for item in base.graph.value_info}
    cand_vi = {item.name: dims(item) for item in cand.graph.value_info}
    vi_diffs = [
        {"name": name, "baseline": base_vi.get(name), "candidate": cand_vi.get(name)}
        for name in sorted(set(base_vi) | set(cand_vi))
        if base_vi.get(name) != cand_vi.get(name)
    ]
    report = {
        "baseline_sha256": sha(BASE),
        "candidate_sha256": sha(CAND),
        "node_differences": node_diffs,
        "initializer_differences": init_diffs,
        "value_info_differences": vi_diffs,
        "graph_input_equal": base.graph.input == cand.graph.input,
        "graph_output_equal": base.graph.output == cand.graph.output,
        "opset_equal": base.opset_import == cand.opset_import,
        "functions_equal": base.functions == cand.functions,
        "clearing_value_info_makes_models_equal": cleared_vi_bytes(base) == cleared_vi_bytes(cand),
        "baseline_value_info_count": len(base.graph.value_info),
        "candidate_value_info_count": len(cand.graph.value_info),
    }
    (HERE / "task105_diff.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
