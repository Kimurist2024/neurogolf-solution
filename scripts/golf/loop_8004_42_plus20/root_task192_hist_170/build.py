#!/usr/bin/env python3
"""Replace task192 color histogram relation with ReduceSum+Slice."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "others/71407/task192.onnx"
OUTPUT = HERE / "task192_hist_slice.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    assert model.graph.node[0].op_type == "Einsum"
    assert model.graph.node[1].op_type == "ArgMax"
    new_nodes = [
        helper.make_node("ReduceSum", ["input", "hist_axes"], ["hist_all"], keepdims=0),
        helper.make_node("Slice", ["hist_all", "hist_start", "hist_end"], ["hist_nonzero"]),
        helper.make_node("ArgMax", ["hist_nonzero"], ["selected_zero_based"], axis=0, keepdims=1),
        helper.make_node("Add", ["selected_zero_based", "hist_one"], ["selected_i64"]),
    ]
    nodes = new_nodes + list(model.graph.node[2:])
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    # color_masks remains live in the final polynomial Einsum; only the
    # histogram selector becomes dead in this variant.
    removed = {"hist_select"}
    initializers = [item for item in model.graph.initializer if item.name not in removed]
    initializers.extend([
        numpy_helper.from_array(np.asarray([0, 2, 3], dtype=np.int64), name="hist_axes"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="hist_start"),
        numpy_helper.from_array(np.asarray([10], dtype=np.int64), name="hist_end"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="hist_one"),
    ])
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)
    before = cost_of(str(SOURCE))
    after = cost_of(str(OUTPUT))
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "before": {"memory": before[0], "params": before[1], "cost": before[2]},
        "after": {"memory": after[0], "params": after[1], "cost": after[2]},
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
