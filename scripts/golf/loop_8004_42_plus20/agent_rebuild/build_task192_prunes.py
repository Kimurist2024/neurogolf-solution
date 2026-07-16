#!/usr/bin/env python3
"""Create honest-shape local-kernel task192 candidates from the LB member.

The incumbent's final 3x5 Conv is the only spatial operation.  These candidates
retain the input-derived color histogram and dynamic color bias, while removing
outer context columns.  No examples or output tables are embedded.
"""

from __future__ import annotations

import copy
import zipfile
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8004.50.zip"


def load_incumbent() -> onnx.ModelProto:
    with zipfile.ZipFile(BASE) as archive:
        return onnx.load_from_string(archive.read("task192.onnx"))


def replace_weight(model: onnx.ModelProto, array) -> None:
    for index, tensor in enumerate(model.graph.initializer):
        if tensor.name == "W":
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(array.astype("float32"), name="W")
            )
            return
    raise KeyError("W")


def main() -> None:
    model = load_incumbent()
    weight = numpy_helper.to_array(next(x for x in model.graph.initializer if x.name == "W"))
    conv = next(node for node in model.graph.node if node.op_type == "Conv")
    candidates = {
        "task192_center3.onnx": (weight[:, :, :, 1:4], [1, 1, 1, 1]),
        "task192_left4.onnx": (weight[:, :, :, :4], [1, 2, 1, 1]),
        "task192_right4.onnx": (weight[:, :, :, 1:], [1, 1, 1, 2]),
    }
    output_dir = HERE / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (new_weight, pads) in candidates.items():
        candidate = copy.deepcopy(model)
        replace_weight(candidate, new_weight)
        candidate_conv = next(node for node in candidate.graph.node if node.op_type == "Conv")
        for attr in candidate_conv.attribute:
            if attr.name == "pads":
                attr.ints[:] = pads
        onnx.checker.check_model(candidate, full_check=True)
        onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
        onnx.save(candidate, output_dir / filename)
        print(filename, list(new_weight.shape), pads)


if __name__ == "__main__":
    main()
