#!/usr/bin/env python3
"""Read-only structural dump for the staged task209 model."""

from __future__ import annotations

import copy
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = ROOT / "others/71407/task209.onnx"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def shape(value: onnx.ValueInfoProto) -> tuple[str, list[int | str]]:
    tensor = value.type.tensor_type
    dims: list[int | str] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return onnx.TensorProto.DataType.Name(tensor.elem_type), dims


def main() -> int:
    model = onnx.load(MODEL)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: shape(value)
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    initializers = {item.name: item for item in model.graph.initializer}
    consumers: dict[str, list[str]] = defaultdict(list)
    for idx, node in enumerate(model.graph.node):
        for name in node.input:
            consumers[name].append(f"{idx}:{node.op_type}")
    print("profile", cost_of(str(MODEL)))
    print("opset", [(item.domain, item.version) for item in model.opset_import])
    print("nodes", len(model.graph.node), "initializers", len(model.graph.initializer))
    print("ops", Counter(node.op_type for node in model.graph.node))
    print("\nINITIALIZERS")
    for item in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(item))
        preview = array.reshape(-1)[:20].tolist()
        print(
            item.name,
            str(array.dtype),
            list(array.shape),
            "n=", array.size,
            "values=", preview,
            "uses=", consumers.get(item.name, []),
        )
    print("\nNODES")
    for idx, node in enumerate(model.graph.node):
        attrs = {}
        for attr in node.attribute:
            try:
                attrs[attr.name] = onnx.helper.get_attribute_value(attr)
            except Exception as exc:  # noqa: BLE001
                attrs[attr.name] = repr(exc)
        outs = [(name, typed.get(name), len(consumers.get(name, []))) for name in node.output]
        print(idx, node.name or "-", node.op_type, "in=", list(node.input), "out=", outs, "attrs=", attrs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
