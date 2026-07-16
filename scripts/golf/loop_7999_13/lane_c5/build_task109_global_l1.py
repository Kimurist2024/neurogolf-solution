#!/usr/bin/env python3
"""Replace task109's spatial ReduceL1 with parameter-free GlobalLpPool.

The incumbent reduces exactly axes H,W with p=1 and keepdims=1.  ONNX
GlobalLpPool(p=1) is the same operation for an NCHW tensor, so the axes
initializer can be removed without changing values, shapes, or runtime memory.
"""

from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "base/task109.onnx"
OUTPUT = HERE / "task109_global_l1.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    users = [
        node
        for node in model.graph.node
        if "reduce_axes" in node.input
    ]
    assert len(users) == 1
    node = users[0]
    assert node.op_type == "ReduceL1"
    assert list(node.input) == ["gn_f16", "reduce_axes"]
    assert list(node.output) == ["counts_avg"]

    node.op_type = "GlobalLpPool"
    del node.input[:]
    node.input.extend(["gn_f16"])
    del node.attribute[:]
    node.attribute.extend([onnx.helper.make_attribute("p", 1)])
    node.name = "counts_l1_global"

    retained = [init for init in model.graph.initializer if init.name != "reduce_axes"]
    assert len(retained) + 1 == len(model.graph.initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)

    model.producer_name = "task109-global-l1-exact-shave"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    # Preserve the incumbent's existing static value_info contract.  Saving
    # the inferred copy rewrites unrelated declarations and changes the
    # measured graph, which is outside this one-operation algebraic shave.
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
