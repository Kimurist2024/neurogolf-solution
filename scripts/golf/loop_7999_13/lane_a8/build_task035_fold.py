#!/usr/bin/env python3
"""Fold task035's duplicated 7x decoder features into ordinary QConv bias."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task035.onnx"
OUTPUT = HERE / "task035_fold_pair_bias.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    graph = model.graph

    removed_outputs = {"core0b", "core1b", "core2b", "core3b", "left2", "right2"}
    kept_nodes = [node for node in graph.node if not removed_outputs.intersection(node.output)]
    del graph.node[:]
    graph.node.extend(kept_nodes)

    kept_initializers = [
        item for item in graph.initializer if item.name not in {"seven_u8", "q_y_scale", "q_w_sp"}
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    # The old decoder saw [x, 7x].  Its positive masks reduce exactly to:
    # ch0 x<=0, ch3 x>=2, ch4 x>=8, ch6 x>=35, ch7 x>=49,
    # ch8 x>=1; all other channels never positive.  Pairing [x, 0]
    # preserves the special QLinearConv padding marker (both padded lanes use
    # x_zero_point=128), while a standard int32 bias encodes each threshold.
    effective_weights = np.array(
        [
            [-1, 0],
            [0, 0],
            [0, 0],
            [1, -1],
            [1, -1],
            [0, 0],
            [1, -1],
            [1, -1],
            [1, -1],
            [0, 0],
        ],
        dtype=np.int16,
    )
    q_weights = (effective_weights + 128).astype(np.uint8).reshape(10, 1, 1, 2)
    bias = np.array([-127, 0, 0, -1, -7, 0, -34, -48, 0, 0], dtype=np.int32)
    graph.initializer.extend(
        [
            numpy_helper.from_array(q_weights, name="q_w_sp"),
            numpy_helper.from_array(bias, name="q_bias"),
        ]
    )

    pack = next(node for node in graph.node if node.output and node.output[0] == "pack")
    zero = "zero_col10"
    del pack.input[:]
    pack.input.extend(
        [
            "left_full", zero,
            zero, zero,
            "core0", zero,
            "core1", zero,
            "core2", zero,
            "core3", zero,
            zero, zero,
            zero, zero,
            zero, zero,
            "right_full", zero,
        ]
    )

    decoder = next(node for node in graph.node if node.output and node.output[0] == "output")
    del decoder.input[:]
    decoder.input.extend(
        ["pack", "one_f", "one_u8", "q_w_sp", "one_f", "one_u8", "one_f", "zero_u8", "q_bias"]
    )

    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, OUTPUT)
    manifest = {
        "source": str(SOURCE.relative_to(HERE.parents[3])),
        "candidate": str(OUTPUT.relative_to(HERE.parents[3])),
        "removed_outputs": sorted(removed_outputs),
        "removed_initializers": ["seven_u8", "q_y_scale"],
        "added_initializer": "q_bias:int32[10]",
        "derivation": "[x,7x] affine threshold -> [x,0] plus standard int32 bias",
    }
    (HERE / "task035_fold_build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
