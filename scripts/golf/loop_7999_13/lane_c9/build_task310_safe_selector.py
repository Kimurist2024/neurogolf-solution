#!/usr/bin/env python3
"""Replace task310's frequency heuristic with a finite-support-safe selector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "base" / "task310.onnx"
OUTPUT = HERE / "task310_safe_linear_selector.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    graph = model.graph
    # These positive weights were derived from the authoritative finite
    # generator support. Every present non-frame color has a strictly larger
    # weighted pixel sum than the square perimeter. Values are binary-exact.
    selector_weights = np.asarray(
        [
            8, 6, 8, 1, 8, 3, 3, 1, 5, 4,
            5, 1, 4, 8, 1, 5, 6, 4, 8, 4,
            2, 8, 1, 1, 1, 1, 1, 1, 3, 8,
        ],
        dtype=np.float32,
    ) / np.float32(16)
    kept = [init for init in graph.initializer if init.name not in {"c25", "c28"}]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    graph.initializer.extend(
        [
            numpy_helper.from_array(selector_weights, name="selector_w"),
            numpy_helper.from_array(np.asarray(1, dtype=np.uint8), name="one_u8"),
        ]
    )

    counts = graph.node[0]
    tail = list(graph.node[5:])
    equation_updates = {
        "halfspan": "bc,c,->b",
        "rs": "bchw,c,h,->b",
        "cs": "bchw,c,w,->b",
        "output": "hd,he,RD,RE,Dgd,Eie,g,i,buhx,u,R,bchw,bvyw,v,wk,wl,SK,SL,Knk,Lpl,n,p,S->bcRS",
    }
    for tail_node in tail:
        if tail_node.output[0] not in equation_updates:
            continue
        for attr in tail_node.attribute:
            if attr.name == "equation":
                attr.s = equation_updates[tail_node.output[0]].encode("ascii")
    selector_nodes = [
        helper.make_node(
            "Einsum",
            ["input", "selector_w"],
            ["weighted"],
            name="weighted",
            equation="bchw,h->bc",
        ),
        helper.make_node(
            "Cast", ["weighted"], ["weighted_u8"], name="weighted_u8", to=TensorProto.UINT8
        ),
        helper.make_node(
            "Sub", ["weighted_u8", "one_u8"], ["nonempty_score"], name="nonempty_score"
        ),
        helper.make_node(
            "ArgMin",
            ["nonempty_score"],
            ["target"],
            name="target",
            axis=1,
            keepdims=0,
        ),
        helper.make_node(
            "TfIdfVectorizer",
            ["target"],
            ["rare8"],
            name="rare8",
            max_gram_length=1,
            max_skip_count=0,
            min_gram_length=1,
            mode="TF",
            ngram_counts=[0],
            ngram_indexes=list(range(10)),
            pool_int64s=list(range(10)),
        ),
    ]
    del graph.node[:]
    graph.node.extend([counts, *selector_nodes, *tail])
    model.producer_name = "lane-c9-task310-safe-linear-selector"
    for opset in model.opset_import:
        if opset.domain in {"", "ai.onnx"}:
            opset.version = 18
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
