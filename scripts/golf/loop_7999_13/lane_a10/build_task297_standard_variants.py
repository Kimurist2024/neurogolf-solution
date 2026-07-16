#!/usr/bin/env python3
"""Build schema-compliant task297 zero-column alternatives for real-cost checks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "baseline" / "task297.onnx"


def trim_weight_and_conv(model: onnx.ModelProto, output: str) -> None:
    init = next(item for item in model.graph.initializer if item.name == "conv_w")
    weights = numpy_helper.to_array(init)
    if weights.shape != (1, 10, 1, 2) or np.count_nonzero(weights[..., 1]) != 0:
        raise RuntimeError("unexpected task297 baseline kernel")
    init.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(weights[..., :1]), "conv_w"))
    node = next(item for item in model.graph.node if item.output[0] == "color_f")
    node.output[0] = output
    del node.attribute[:]
    node.attribute.extend(
        [
            helper.make_attribute("dilations", [1, 1]),
            helper.make_attribute("strides", [30, 1]),
            helper.make_attribute("pads", [0, 0, 0, 0]),
        ]
    )


def save(model: onnx.ModelProto, name: str, detail: str) -> dict[str, object]:
    path = HERE / name
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "detail": detail,
    }


def slice_after_conv(base: onnx.ModelProto) -> dict[str, object]:
    model = copy.deepcopy(base)
    trim_weight_and_conv(model, "color_full_f")
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), "crop_start"),
            numpy_helper.from_array(np.asarray([6], dtype=np.int64), "crop_end"),
            numpy_helper.from_array(np.asarray([3], dtype=np.int64), "crop_axis"),
        ]
    )
    model.graph.node.insert(
        1,
        helper.make_node(
            "Slice",
            ["color_full_f", "crop_start", "crop_end", "crop_axis"],
            ["color_f"],
            name="standard_crop_first_six",
        ),
    )
    return save(
        model,
        "task297_standard_slice.onnx",
        "1-column Conv with non-negative pads, followed by explicit Slice width 0:6",
    )


def split_after_cast(base: onnx.ModelProto) -> dict[str, object]:
    model = copy.deepcopy(base)
    trim_weight_and_conv(model, "color_full_f")
    cast = next(item for item in model.graph.node if item.output[0] == "color_u8")
    cast.input[0] = "color_full_f"
    cast.output[0] = "color_full_u8"
    sign = next(item for item in model.graph.node if item.output[0] == "active_u8")
    sign_copy = copy.deepcopy(sign)
    sign_copy.input[0] = "color_u8"
    del model.graph.node[next(index for index, item in enumerate(model.graph.node) if item is sign)]
    split = next(item for item in model.graph.node if item.op_type == "Split")
    split.input[0] = "color_full_u8"
    del split.output[:]
    split.output.extend([f"c{index}" for index in range(30)])
    for attr in split.attribute:
        if attr.name == "num_outputs":
            attr.i = 30
    split_index = next(index for index, item in enumerate(model.graph.node) if item is split)
    model.graph.node.insert(
        split_index + 1,
        helper.make_node("Concat", [f"c{i}" for i in range(6)], ["color_u8"], axis=3),
    )
    model.graph.node.insert(split_index + 2, sign_copy)
    return save(
        model,
        "task297_standard_split.onnx",
        "1-column Conv with non-negative pads, Split 30 columns, re-concat first six",
    )


def main() -> None:
    base = onnx.load(SOURCE)
    rows = [slice_after_conv(base), split_after_cast(base)]
    (HERE / "task297_standard_build.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
