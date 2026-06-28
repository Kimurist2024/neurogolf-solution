#!/usr/bin/env python3
"""Build a minimal static-shape ONNX for task030."""
from __future__ import annotations

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto


def make_const(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return helper.make_tensor(name, helper.np_dtype_to_tensor_dtype(arr.dtype), arr.shape, arr.tobytes(), raw=True)


def build() -> onnx.ModelProto:
    H, W = 10, 10  # internal working grid size (task030 grids fit here)
    Hout, Wout = 30, 30

    # Build DIFF[i, j] = i - j for vertical shift selection
    diff = np.arange(H, dtype=np.int64)[:, None] - np.arange(H, dtype=np.int64)[None, :]

    # Slice starts/ends for axes [1, 2, 3]
    def slice_consts(channel: int):
        return {
            f"st{channel}": np.array([channel, 0, 0], dtype=np.int64),
            f"en{channel}": np.array([channel + 1, H, W], dtype=np.int64),
        }

    initializers = []
    initializers.append(make_const("DIFF", diff))
    initializers.append(make_const("axes3", np.array([1, 2, 3], dtype=np.int64)))
    for c in (0, 1, 2, 4):
        initializers.extend(make_const(k, v) for k, v in slice_consts(c).items())

    # Pad: [1, 5, 10, 10] bool -> [1, 10, 30, 30] bool
    # ONNX Pad pads format [begin_1, end_1, begin_2, end_2, ...]
    pads = np.array([0, 0, 0, 0, 0, 5, 20, 20], dtype=np.int64)
    initializers.append(make_const("pads30", pads))
    initializers.append(helper.make_tensor("ZERO_BOOL", TensorProto.BOOL, [], [False]))
    initializers.append(make_const("zero_ch", np.zeros((1, 1, H, W), dtype=bool)))

    nodes = []

    # Slice channels 0 (background) and colors 1, 2, 4
    for c in (0, 1, 2, 4):
        nodes.append(helper.make_node(
            "Slice",
            inputs=["input", f"st{c}", f"en{c}", "axes3"],
            outputs=[f"ch{c}"],
        ))

    # Color 1: bool mask + top row
    nodes.append(helper.make_node("Cast", inputs=["ch1"], outputs=["ch1b"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("ReduceMax", inputs=["ch1"], outputs=["rh1"], axes=[3], keepdims=1))
    nodes.append(helper.make_node("ArgMax", inputs=["rh1"], outputs=["top1"], axis=2, keepdims=1, select_last_index=0))

    # Colors 2, 4: float16 mask + top row
    for c in (2, 4):
        nodes.append(helper.make_node("ReduceMax", inputs=[f"ch{c}"], outputs=[f"rh{c}"], axes=[3], keepdims=1))
        nodes.append(helper.make_node("ArgMax", inputs=[f"rh{c}"], outputs=[f"top{c}"], axis=2, keepdims=1, select_last_index=0))
        nodes.append(helper.make_node("Cast", inputs=[f"ch{c}"], outputs=[f"ch{c}f"], to=TensorProto.FLOAT16))

    # Compute shifts: delta_c = top_1 - top_c  (positive -> shift down)
    for c in (2, 4):
        nodes.append(helper.make_node("Sub", inputs=["top1", f"top{c}"], outputs=[f"delta{c}"]))

    # Build shift selectors and apply MatMul shift for colors 2, 4
    for c in (2, 4):
        nodes.append(helper.make_node("Equal", inputs=["DIFF", f"delta{c}"], outputs=[f"Sb{c}"]))
        nodes.append(helper.make_node("Cast", inputs=[f"Sb{c}"], outputs=[f"Sf{c}"], to=TensorProto.FLOAT16))
        # MatMul: Sf (10,10) @ ch_f (1,1,10,10) -> (1,1,10,10)
        nodes.append(helper.make_node("MatMul", inputs=[f"Sf{c}", f"ch{c}f"], outputs=[f"outp{c}f"]))
        nodes.append(helper.make_node("Cast", inputs=[f"outp{c}f"], outputs=[f"outp{c}"], to=TensorProto.BOOL))

    # Background channel 0: (input_background OR original_color2 OR original_color4) XOR (shifted_color2 OR shifted_color4)
    nodes.append(helper.make_node("Cast", inputs=["ch0"], outputs=["ch0b"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", inputs=["ch2f"], outputs=["ch2b"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", inputs=["ch4f"], outputs=["ch4b"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Or", inputs=["ch0b", "ch2b"], outputs=["tmp_base"]))
    nodes.append(helper.make_node("Or", inputs=["tmp_base", "ch4b"], outputs=["base"]))
    nodes.append(helper.make_node("Or", inputs=["outp2", "outp4"], outputs=["paint"]))
    nodes.append(helper.make_node("Xor", inputs=["base", "paint"], outputs=["out0"]))

    # Concatenate output channels [0, 1, 2, 3(placeholder), 4]
    nodes.append(helper.make_node(
        "Concat",
        inputs=["out0", "ch1b", "outp2", "zero_ch", "outp4"],
        outputs=["out5"],
        axis=1,
    ))

    # Pad to final size [1, 10, 30, 30]
    nodes.append(helper.make_node("Pad", inputs=["out5", "pads30", "ZERO_BOOL"], outputs=["output"], mode="constant"))

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, Hout, Wout])]
    outputs = [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, Hout, Wout])]

    graph = helper.make_graph(nodes, "task030", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    model.producer_name = "task030_builder"
    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = build()
    onnx.save(model, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
