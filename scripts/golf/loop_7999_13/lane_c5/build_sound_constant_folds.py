#!/usr/bin/env python3
"""Constant-fold exact shape scaffolding in the C5 sound incumbents.

These rewrites do not infer any task output from examples.  They replace
Shape/ConstantOfShape results whose inputs are fixed initializers, and remove
CenterCropPad calls that provably map a tensor to its existing shape.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def add_i64(model: onnx.ModelProto, name: str, values: list[int]) -> None:
    assert all(init.name != name for init in model.graph.initializer)
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)
    )


def remove_nodes_by_output(model: onnx.ModelProto, outputs: set[str]) -> None:
    removed = [node for node in model.graph.node if set(node.output) & outputs]
    assert {name for node in removed for name in node.output if name in outputs} == outputs
    kept = [node for node in model.graph.node if not (set(node.output) & outputs)]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def rewire(model: onnx.ModelProto, old: str, new: str) -> None:
    uses = 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
                uses += 1
    assert uses > 0


def validate_and_save(model: onnx.ModelProto, output: Path, producer: str) -> None:
    model.producer_name = producer
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    # Preserve unrelated incumbent value_info declarations.  The executable
    # graph changes here are fully checked on a separate inferred copy.
    onnx.save(model, output)


def build109() -> None:
    model = onnx.load(HERE / "base/task109.onnx")
    # Shape(j_i8[12]) == int64 vector [12].
    remove_nodes_by_output(model, {"shape7_dyn"})
    add_i64(model, "shape7_dyn", [12])
    validate_and_save(model, HERE / "task109_shape_fold.onnx", "task109-exact-shape-fold")


def build112() -> None:
    model = onnx.load(HERE / "base/task112.onnx")
    # All four outputs are one-element int64 shape vectors with fixed values.
    fixed = {
        "s30_hid": [30],
        "input_shape_values_shape": [4],
        "indices5_shape_dyn": [5],
        "s38_hid": [38],
    }
    remove_nodes_by_output(model, set(fixed))
    for name, values in fixed.items():
        add_i64(model, name, values)

    # input is canonically [1,10,30,30], so cropping/padding only H,W to 30
    # is an identity.  Both consumers can use the free graph input directly.
    rewire(model, "input_cloak30", "input")
    remove_nodes_by_output(model, {"input_cloak30"})
    validate_and_save(model, HERE / "task112_shape_fold.onnx", "task112-exact-shape-fold")


def build170() -> None:
    model = onnx.load(HERE / "base/task170.onnx")
    # Shape(slice_axes123_i32[3]) == int64 vector [3].
    remove_nodes_by_output(model, {"shape3_i64"})
    add_i64(model, "shape3_i64", [3])
    validate_and_save(model, HERE / "task170_shape_fold.onnx", "task170-exact-shape-fold")


def build245() -> None:
    model = onnx.load(HERE / "base/task245.onnx")
    # Shape(input_shape_values[4]) and CastLike(int8[5], int64 reference).
    remove_nodes_by_output(model, {"shape4_dyn", "five_shape_dyn"})
    add_i64(model, "shape4_dyn", [4])
    add_i64(model, "five_shape_dyn", [5])
    retained = [init for init in model.graph.initializer if init.name != "five_i8"]
    assert len(retained) + 1 == len(model.graph.initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)

    # CenterCropPad(input_shape_values, [4]) is the identity, and using that
    # result to resize the canonical input to [1,10,30,30] is also identity.
    rewire(model, "input_cloak", "input")
    remove_nodes_by_output(model, {"input_shape_dyn_hidden", "input_cloak"})
    validate_and_save(model, HERE / "task245_shape_fold.onnx", "task245-exact-shape-fold")


def main() -> None:
    build109()
    build112()
    build170()
    build245()


if __name__ == "__main__":
    main()
