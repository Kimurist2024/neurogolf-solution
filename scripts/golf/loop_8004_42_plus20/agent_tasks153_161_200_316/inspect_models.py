from __future__ import annotations

from pathlib import Path

import onnx
from onnx import numpy_helper


BASE = Path(__file__).resolve().parent / "baseline"


for task in (153, 161, 200, 316):
    path = BASE / f"task{task:03d}.onnx"
    model = onnx.load(path)
    graph = model.graph
    print(f"\n=== task{task:03d} ===")
    print("opsets", [(x.domain, x.version) for x in model.opset_import])
    print("inputs", [(x.name, str(x.type)) for x in graph.input])
    print("outputs", [(x.name, str(x.type)) for x in graph.output])
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        print("INIT", init.name, arr.dtype, arr.shape, arr.tolist())
    for i, node in enumerate(graph.node):
        attrs = []
        for attr in node.attribute:
            value = onnx.helper.get_attribute_value(attr)
            if hasattr(value, "tolist"):
                value = value.tolist()
            attrs.append((attr.name, value))
        print(i, node.op_type, list(node.input), "->", list(node.output), attrs)
