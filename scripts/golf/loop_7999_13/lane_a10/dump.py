#!/usr/bin/env python3
"""Write exact A10 initializer, value-info, and node dumps."""

from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


for path in sorted((HERE / "baseline").glob("task*.onnx")):
    model = onnx.load(path)
    lines = [f"# {path.name}", "", "## Initializers"]
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        text = repr(array.tolist())
        if len(text) > 5000:
            text = text[:5000] + "..."
        lines.append(f"{init.name}: {array.dtype} {array.shape} {text}")
    lines.extend(["", "## Value info"])
    for item in model.graph.value_info:
        tensor = item.type.tensor_type
        shape = [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            for dim in tensor.shape.dim
        ]
        lines.append(f"{item.name}: {onnx.TensorProto.DataType.Name(tensor.elem_type)} {shape}")
    lines.extend(["", "## Nodes"])
    for index, node in enumerate(model.graph.node):
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        lines.append(
            f"{index:02d} {node.op_type} {list(node.input)} -> {list(node.output)} {attrs}"
        )
    (HERE / f"{path.stem}_dump.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
