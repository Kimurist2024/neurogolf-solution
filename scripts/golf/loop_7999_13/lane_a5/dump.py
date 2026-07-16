#!/usr/bin/env python3
"""Dump A5 nodes and initializer values for exact algebra review."""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def attr_value(attr):
    value = onnx.helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, onnx.TensorProto):
        return numpy_helper.to_array(value).tolist()
    return value


def main() -> None:
    for path in sorted((HERE / "baseline").glob("task*.onnx")):
        model = onnx.load(path)
        lines = [f"# {path.name}", "", "## Initializers"]
        for init in model.graph.initializer:
            array = numpy_helper.to_array(init)
            lines.append(f"{init.name}: {array.dtype} {array.shape} {array.tolist()}")
        lines += ["", "## Nodes"]
        for index, node in enumerate(model.graph.node):
            attrs = {attr.name: attr_value(attr) for attr in node.attribute}
            lines.append(f"{index:02d} {node.op_type} {list(node.input)} -> {list(node.output)} {attrs}")
        (HERE / f"{path.stem}_dump.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
