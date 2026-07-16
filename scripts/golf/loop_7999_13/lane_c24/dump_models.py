#!/usr/bin/env python3
"""Print exact C24 model structure for manual soundness and shave review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    tensor = value_info.type.tensor_type
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in tensor.shape.dim
    ]


def main() -> None:
    for task in (363, 388):
        path = HERE / "base" / f"task{task}.onnx"
        model = onnx.load(path)
        print(f"\n## task{task}: {path}")
        print("opsets", [(entry.domain, entry.version) for entry in model.opset_import])
        print("inputs", [(item.name, shape(item)) for item in model.graph.input])
        print("outputs", [(item.name, shape(item)) for item in model.graph.output])
        print("value_info", [(item.name, shape(item)) for item in model.graph.value_info])
        print("initializers")
        for item in model.graph.initializer:
            value = numpy_helper.to_array(item)
            flat = value.reshape(-1)
            shown = np.array2string(flat[:80], threshold=80, separator=",")
            print(
                f"  {item.name}: dtype={value.dtype} shape={list(value.shape)} "
                f"count={value.size} values={shown}"
            )
        print("nodes")
        for index, node in enumerate(model.graph.node):
            attributes = []
            for attr in node.attribute:
                rendered = onnx.helper.get_attribute_value(attr)
                if isinstance(rendered, bytes):
                    rendered = rendered.decode("utf-8", errors="replace")
                attributes.append(f"{attr.name}={rendered}")
            print(
                f"  {index:02d} {node.name!r} {node.domain!r}:{node.op_type} "
                f"in={list(node.input)} out={list(node.output)} attrs={attributes}"
            )


if __name__ == "__main__":
    main()
