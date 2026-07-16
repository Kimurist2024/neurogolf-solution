#!/usr/bin/env python3
"""Replace task077's three-input floating Einsum packer with an honest Conv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph
    nodes = list(graph.node)
    if [node.op_type for node in nodes[:4]] != ["Einsum", "Reshape", "Slice", "Cast"]:
        raise RuntimeError("unexpected task077 exact packer prefix")
    if nodes[0].output[0] != "packed" or nodes[1].output[0] != "packed4":
        raise RuntimeError("unexpected packer tensor names")

    weights = np.zeros((1, 10, 1, 30), dtype=np.float32)
    weights[0, 2, 0, :] = 2.0 ** np.arange(30, dtype=np.float32)
    conv_weight = numpy_helper.from_array(weights, "pack_conv_weight")
    conv = helper.make_node(
        "Conv",
        ["input", "pack_conv_weight"],
        ["packed4"],
        name="packed4",
        kernel_shape=[1, 30],
        strides=[1, 1],
    )
    del graph.node[:]
    graph.node.extend([conv, *nodes[2:]])
    kept = [item for item in graph.initializer if item.name not in {"chan2", "pow30", "shp30"}]
    del graph.initializer[:]
    graph.initializer.extend([*kept, conv_weight])

    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
