#!/usr/bin/env python3
"""Remove task349's hstart lookup using hstart = 1 - 3 * radius."""

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
SOURCE = ROOT / "others" / "71407" / "task349.onnx"
OUTPUT = HERE / "task349_affine_hstart.onnx"


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    model = onnx.load(SOURCE)
    arrays = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    hstart = arrays["hstart_offset_by_mod_i8"].astype(np.int16)
    radius = arrays["hend_offset_by_mod_i8"].astype(np.int16)
    assert np.array_equal(hstart, 1 - 3 * radius)
    assert hstart.min() >= -128 and hstart.max() <= 127

    kept = [x for x in model.graph.initializer if x.name != "hstart_offset_by_mod_i8"]
    kept.append(numpy_helper.from_array(np.asarray(-3, dtype=np.int8), "neg3_i8"))
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    nodes = []
    removed = 0
    for node in model.graph.node:
        if node.output and node.output[0] == "hstart_offset_i8":
            assert node.op_type == "Gather"
            assert list(node.input) == ["hstart_offset_by_mod_i8", "radius_code"]
            removed += 1
            continue
        nodes.append(node)
        if node.output and node.output[0] == "hend_offset_i8":
            nodes.append(
                helper.make_node(
                    "Mul",
                    ["hend_offset_i8", "neg3_i8"],
                    ["hstart_minus_one_i8"],
                    name="hstart_minus_one_from_radius",
                )
            )
            nodes.append(
                helper.make_node(
                    "Add",
                    ["hstart_minus_one_i8", "one_i8"],
                    ["hstart_offset_i8"],
                    name="hstart_from_radius_affine",
                )
            )
    assert removed == 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)

    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "table_identity": {
            "hstart": hstart.tolist(),
            "radius": radius.tolist(),
            "all_equal_1_minus_3r": True,
        },
        "source_profile": profile(SOURCE),
        "candidate_profile": profile(OUTPUT),
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
