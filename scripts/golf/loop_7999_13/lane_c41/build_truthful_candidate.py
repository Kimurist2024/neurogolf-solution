#!/usr/bin/env python3
"""Build a truthful-shape task204 rewrite from the exact 8002.63 member.

The incumbent casts/crops the complete input tensor before bit packing.  This
rewrite bit-packs the original float input first (generator widths are <= 20,
so every resulting integer is exactly representable by float32), then casts
only the compact packed rows.  The final reconstruction is likewise performed
in float32, allowing the original input to feed the output directly.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task204.onnx"
OUTPUT = HERE / "candidates" / "task204_truthful_floatpack.onnx"


def replace_initializer_dtype(model: onnx.ModelProto, name: str) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name != name:
            continue
        array = numpy_helper.to_array(initializer).astype(np.float32)
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name))
        return
    raise KeyError(name)


def main() -> None:
    model = onnx.load(SOURCE)
    graph = model.graph

    # Float inputs to the two Einsums avoid full-grid integer intermediates.
    for name in ("sel1", "pack_powers", "input_sel2", "final_coeff", "shift_unpack_consts"):
        replace_initializer_dtype(model, name)

    remove = {
        "input_full_hide",
        "input_full_i32",
        "input_h29",
        "input_h28",
        "input_h27",
        "input_h26",
        "input_h25",
        "input_h24",
        "input_h23",
        "input_h22",
        "input_h21",
        "input_h20",
    }
    old_nodes = [copy.deepcopy(node) for node in graph.node if node.name not in remove]
    del graph.node[:]

    for node in old_nodes:
        if node.name == "blue_30":
            node.input[0] = "input"
            graph.node.append(node)
            graph.node.append(
                helper.make_node(
                    "CenterCropPad",
                    ["blue_30", "__ng_row_e20_dyn"],
                    ["blue_20_float"],
                    name="blue_20_float",
                    axes=[1],
                )
            )
            graph.node.append(
                helper.make_node(
                    "CastLike",
                    ["blue_20_float", "div8"],
                    ["blue_20"],
                    name="blue_20",
                )
            )
            continue

        for index, value in enumerate(node.input):
            if value == "blue_30":
                node.input[index] = "blue_20"

        if node.name == "output":
            node.input[0] = "input"
            node.input[2] = "masks5_rows30_float"
            graph.node.append(
                helper.make_node(
                    "CastLike",
                    ["masks5_rows30", "input"],
                    ["masks5_rows30_float"],
                    name="masks5_rows30_float",
                )
            )
        graph.node.append(node)

    # The actual output always has the benchmark tensor shape.
    graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    dims = graph.output[0].type.tensor_type.shape.dim
    for dim, value in zip(dims, (1, 10, 30, 30)):
        dim.Clear()
        dim.dim_value = value

    # Re-infer instead of retaining any incumbent value_info declarations.
    del graph.value_info[:]
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

    row18 = {
        "blue_rows",
        "blue_up19",
        "blue_down",
        "vertical_sides",
        "p6_shift",
        "p6",
        "p48_shift",
        "p48",
        "p3072_shift",
        "p3072",
        "px16_shift",
        "px16",
        "px_xor_v",
        "interior_rows",
        "px_xor_even",
        "px_xor_even_plus_even",
        "same_parity_rights",
        "odd_right_boundary",
        "g2_mul",
        "odd_g2",
        "qshift3",
        "gshift4",
        "gadd4",
        "orange_rows",
    }
    shapes: dict[str, tuple[int, ...]] = {
        "blue_30": (1, 30, 1),
        "blue_20_float": (1, 20, 1),
        "blue_20": (1, 20, 1),
        "blue_up19": (1, 19, 1),
        "orange_20": (1, 20, 1),
        "interior_20": (1, 20, 1),
        "ones20_bool": (1, 20, 1),
        "ones20_i32": (1, 20, 1),
        "masks5_20": (3, 20, 1),
        "masks5_rows30_float": (3, 30, 1),
    }
    shapes.update({name: (1, 18, 1) for name in row18 if name != "blue_up19"})
    shapes.update({f"masks5_rows{rows}": (3, rows, 1) for rows in range(21, 31)})
    for value in inferred.graph.value_info:
        shape = shapes.get(value.name)
        if shape is None:
            continue
        del value.type.tensor_type.shape.dim[:]
        for extent in shape:
            value.type.tensor_type.shape.dim.add().dim_value = extent

    onnx.checker.check_model(inferred, full_check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
