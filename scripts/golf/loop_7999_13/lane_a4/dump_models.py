#!/usr/bin/env python3
"""Human-readable model dump for algebra review."""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def attrs(node: onnx.NodeProto) -> dict[str, object]:
    out = {}
    for attr in node.attribute:
        value = onnx.helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, onnx.TensorProto):
            value = numpy_helper.to_array(value).tolist()
        out[attr.name] = value
    return out


def main() -> None:
    for path in sorted((HERE / "baseline").glob("task*.onnx")):
        model = onnx.load(path)
        lines = [f"# {path.name}", "", "## I/O"]
        for value in list(model.graph.input) + list(model.graph.output):
            lines.append(str(value).strip())
        lines.extend(["", "## Initializers"])
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            lines.append(
                f"{init.name}: dtype={arr.dtype} shape={arr.shape} value={arr.tolist()}"
            )
        lines.extend(["", "## Nodes"])
        for index, node in enumerate(model.graph.node):
            lines.append(
                f"{index:02d} {node.op_type} {list(node.input)} -> {list(node.output)} attrs={attrs(node)}"
            )
        (HERE / f"{path.stem}_dump.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
