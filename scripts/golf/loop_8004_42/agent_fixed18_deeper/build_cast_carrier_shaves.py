#!/usr/bin/env python3
"""Replace initializer-backed CastLike nodes with exact Cast nodes.

When an initializer is used *only* as CastLike's type carrier, its value and
shape are semantically irrelevant.  Replacing every such CastLike with Cast
preserves values exactly while deleting the carrier parameter.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "current"
OUT_DIR = HERE / "candidates"
TASKS = (31, 71, 88, 302)


def build(task: int) -> Path:
    model = onnx.load(str(SOURCE_DIR / f"task{task:03d}.onnx"))
    initializers = {init.name: init for init in model.graph.initializer}
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            consumers[name].append((node, index))

    carriers = {
        name
        for name in initializers
        if consumers[name]
        and all(node.op_type == "CastLike" and index == 1 for node, index in consumers[name])
    }
    if not carriers:
        raise RuntimeError(f"task{task:03d}: no initializer-backed CastLike carrier")

    replacement_nodes: list[onnx.NodeProto] = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type == "CastLike" and len(node.input) == 2 and node.input[1] in carriers:
            dtype = initializers[node.input[1]].data_type
            replacement_nodes.append(
                helper.make_node(
                    "Cast",
                    [node.input[0]],
                    list(node.output),
                    name=node.name,
                    to=dtype,
                )
            )
            replaced += 1
        else:
            replacement_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(replacement_nodes)

    kept_initializers = [init for init in model.graph.initializer if init.name not in carriers]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"task{task:03d}_cast_carrier.onnx"
    onnx.save(model, str(output))
    print(
        f"task{task:03d}: carriers={sorted(carriers)} replaced={replaced} -> {output}"
    )
    return output


def main() -> int:
    for task in TASKS:
        build(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
