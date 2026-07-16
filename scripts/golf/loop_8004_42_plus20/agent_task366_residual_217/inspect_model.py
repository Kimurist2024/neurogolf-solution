#!/usr/bin/env python3
"""Inspect the current staged task366 graph for exact residual golf leads."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = ROOT / "others/71407/task366.onnx"


def dims(value: onnx.ValueInfoProto) -> tuple[int | None, ...]:
    return tuple(
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    )


def main() -> None:
    model = onnx.load(MODEL)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    consumers: dict[str, list[str]] = defaultdict(list)
    producer: dict[str, str] = {}
    for index, node in enumerate(model.graph.node):
        label = f"{index}:{node.op_type}:{node.name}"
        for name in node.input:
            consumers[name].append(label)
        for name in node.output:
            producer[name] = label

    print("opset", [(item.domain, item.version) for item in model.opset_import])
    print("nodes", len(model.graph.node), Counter(node.op_type for node in model.graph.node))
    print("initializers", len(initializers), "params", sum(array.size for array in initializers.values()))
    print("graph outputs", [value.name for value in model.graph.output])
    print("\nINITIALIZERS")
    for name, array in sorted(initializers.items(), key=lambda item: (-item[1].size, item[0])):
        flat = array.reshape(-1)
        unique = np.unique(flat)
        summary = (
            unique.tolist()
            if unique.size <= 12
            else {"min": str(unique[0]), "max": str(unique[-1]), "unique": int(unique.size)}
        )
        print(name, array.dtype, array.shape, "size", array.size, "unique", summary, "uses", consumers[name])

    print("\nDUPLICATE INITIALIZER CONTENT")
    groups: dict[tuple[str, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for name, array in initializers.items():
        groups[(array.dtype.str, array.shape, array.tobytes())].append(name)
    for names in groups.values():
        if len(names) > 1:
            print(names)

    print("\nIDENTITY-CANDIDATE NODES")
    dtype_by_name = {
        name: value.type.tensor_type.elem_type for name, value in typed.items()
    }
    for index, node in enumerate(model.graph.node):
        reason = None
        if node.op_type == "Identity":
            reason = "Identity"
        elif node.op_type == "Cast" and node.input[0] in dtype_by_name:
            target = next(
                int(onnx.helper.get_attribute_value(attr))
                for attr in node.attribute
                if attr.name == "to"
            )
            if dtype_by_name[node.input[0]] == target:
                reason = "same-dtype Cast"
        elif (
            node.op_type == "CastLike"
            and node.input[0] in dtype_by_name
            and node.input[1] in dtype_by_name
            and dtype_by_name[node.input[0]] == dtype_by_name[node.input[1]]
        ):
            reason = "same-dtype CastLike"
        elif node.op_type in {"Mul", "Div", "Add", "Sub"}:
            for input_index, name in enumerate(node.input):
                array = initializers.get(name)
                if array is None or array.size != 1:
                    continue
                value = array.item()
                if node.op_type == "Mul" and value == 1:
                    reason = f"Mul one input{input_index}"
                elif node.op_type == "Div" and input_index == 1 and value == 1:
                    reason = "Div one"
                elif node.op_type == "Add" and value == 0:
                    reason = f"Add zero input{input_index}"
                elif node.op_type == "Sub" and input_index == 1 and value == 0:
                    reason = "Sub zero"
        if reason:
            print(index, node.op_type, node.name, reason, list(node.input), list(node.output))

    print("\nOUTPUT MEMORY RANK")
    dtype_bytes = {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT16: 2,
        onnx.TensorProto.INT16: 2,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.BOOL: 1,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.UINT32: 4,
        onnx.TensorProto.UINT64: 8,
    }
    ranked = []
    graph_outputs = {value.name for value in model.graph.output}
    for index, node in enumerate(model.graph.node):
        for name in node.output:
            if name in graph_outputs or name not in typed:
                continue
            shape = dims(typed[name])
            if None in shape:
                continue
            size = int(np.prod(shape, dtype=np.int64)) * dtype_bytes[dtype_by_name[name]]
            ranked.append((size, index, node.op_type, node.name, name, shape, len(consumers[name])))
    for row in sorted(ranked, reverse=True)[:100]:
        print(row)

    print("\nEXACT DUPLICATE NODE SIGNATURES")
    signatures: dict[tuple[object, ...], list[tuple[int, str, list[str]]]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        attrs = tuple(sorted(attr.SerializeToString() for attr in node.attribute))
        signature = (node.domain, node.op_type, tuple(node.input), attrs, len(node.output))
        signatures[signature].append((index, node.name, list(node.output)))
    for rows in signatures.values():
        if len(rows) > 1:
            print(rows)

    print("\nROUND CHAINS")
    by_output = {name: (index, node) for index, node in enumerate(model.graph.node) for name in node.output}
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Round":
            continue
        parent = by_output.get(node.input[0])
        children = [label for label in consumers[node.output[0]]]
        print(index, node.name, "parent", parent[0:1] if parent else None, parent[1].op_type if parent else None,
              "input", list(node.input), "output", list(node.output), "children", children)

    print("\nLOG ANCESTORS")
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Log":
            continue
        chain = []
        frontier = [node.input[0]]
        seen = set()
        for _ in range(5):
            next_frontier = []
            for name in frontier:
                if name in seen or name not in by_output:
                    continue
                seen.add(name)
                producer_index, producer_node = by_output[name]
                chain.append((producer_index, producer_node.op_type, producer_node.name,
                              list(producer_node.input), list(producer_node.output)))
                next_frontier.extend(producer_node.input)
            frontier = next_frontier
        print(index, node.name, "input", node.input[0], "ancestors", sorted(chain))

    print("\nNODE DETAIL")
    for index, node in enumerate(model.graph.node):
        outputs = [
            f"{name}:{typed[name].type.tensor_type.elem_type}{dims(typed[name])}"
            if name in typed
            else name
            for name in node.output
        ]
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        print(index, node.op_type, node.name, "in", list(node.input), "out", outputs, "attr", attrs)


if __name__ == "__main__":
    main()
