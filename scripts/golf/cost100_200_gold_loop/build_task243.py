#!/usr/bin/env python3
"""Remove task243's fixed-shape Shape/Reshape identity chain exactly."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"
OUTPUT = HERE / "candidates/task243_shape_reshape_exact.onnx"
EVIDENCE = HERE / "task243_build.json"


def main() -> None:
    with zipfile.ZipFile(AUTHORITY) as archive:
        source = archive.read("task243.onnx")
    model = onnx.load_model_from_string(source)

    if [node.op_type for node in model.graph.node[:2]] != ["Shape", "Reshape"]:
        raise RuntimeError("unexpected task243 authority prefix")
    if list(model.graph.node[0].input) != ["r0"]:
        raise RuntimeError("Shape no longer reads r0")
    if list(model.graph.node[1].input) != ["r0", "shape_r"]:
        raise RuntimeError("Reshape no longer has the fixed identity form")
    r0 = next(item for item in model.graph.initializer if item.name == "r0")
    if list(r0.dims) != [30]:
        raise RuntimeError("r0 is no longer statically length 30")

    # The Reshape output is exactly r0 because its requested shape is Shape(r0).
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "r_u":
                node.input[index] = "r0"

    # The only remaining Shape result use is Slice.ends; encode the proven
    # initializer length directly as one scalar parameter.
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([30], dtype=np.int64), "end30")
    )
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "shape_r":
                node.input[index] = "end30"

    del model.graph.node[:2]
    kept_info = [
        value
        for value in model.graph.value_info
        if value.name not in {"shape_r", "r_u"}
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_info)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    EVIDENCE.write_text(
        json.dumps(
            {
                "authority": AUTHORITY.name,
                "authority_member_sha256": hashlib.sha256(source).hexdigest(),
                "candidate": str(OUTPUT.relative_to(ROOT)),
                "candidate_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                "proof": [
                    "r0 has immutable static shape [30]",
                    "Reshape(r0, Shape(r0)) is identity",
                    "after identity removal Shape(r0) only supplied Slice.ends",
                    "Slice.ends is replaced by int64 initializer [30]",
                ],
                "removed_nodes": ["Shape", "Reshape"],
                "added_initializers": {"end30": [30]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
