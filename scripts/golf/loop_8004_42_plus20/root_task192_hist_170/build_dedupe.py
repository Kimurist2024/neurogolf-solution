#!/usr/bin/env python3
"""Deduplicate task192's identical [0,1] histogram and OneHot constants."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "others/71407/task192.onnx"
OUTPUT = HERE / "task192_dedupe_values.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    assert np.array_equal(values["hist_select"], values["onehot_values"])
    rewired = 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "onehot_values":
                node.input[index] = "hist_select"
                rewired += 1
    assert rewired == 1
    keep = [item for item in model.graph.initializer if item.name != "onehot_values"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)
    before = cost_of(str(SOURCE))
    after = cost_of(str(OUTPUT))
    result = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "before": {"memory": before[0], "params": before[1], "cost": before[2]},
        "after": {"memory": after[0], "params": after[1], "cost": after[2]},
        "proof": "hist_select and onehot_values are byte-identical float32 [0,1] tensors; all uses see the same tensor value and shape",
    }
    (HERE / "dedupe_build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
