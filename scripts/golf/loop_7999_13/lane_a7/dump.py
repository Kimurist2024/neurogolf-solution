#!/usr/bin/env python3
"""Write compact textual dumps for the isolated A7 members."""

from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def main() -> None:
    for path in sorted((HERE / "baseline").glob("task*.onnx")):
        model = onnx.load(path)
        lines = [f"# {path.name}", "", "## Initializers"]
        for init in model.graph.initializer:
            array = numpy_helper.to_array(init)
            value = repr(array.tolist())
            if len(value) > 1000:
                value = value[:1000] + "..."
            lines.append(f"{init.name}: {array.dtype} {array.shape} {value}")
        lines.extend(["", "## Value info"])
        for vi in model.graph.value_info:
            tensor = vi.type.tensor_type
            shape = [d.dim_value if d.HasField("dim_value") else "?" for d in tensor.shape.dim]
            lines.append(f"{vi.name}: {onnx.TensorProto.DataType.Name(tensor.elem_type)} {shape}")
        lines.extend(["", "## Nodes"])
        for index, node in enumerate(model.graph.node):
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            if node.op_type == "TfIdfVectorizer":
                attrs = {key: len(value) if isinstance(value, list) else value for key, value in attrs.items()}
            lines.append(f"{index:02d} {node.op_type} {list(node.input)} -> {list(node.output)} {attrs}")
        (HERE / f"{path.stem}_dump.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
