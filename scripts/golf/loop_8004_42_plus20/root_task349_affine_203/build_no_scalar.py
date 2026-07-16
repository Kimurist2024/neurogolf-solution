#!/usr/bin/env python3
"""Remove task349's hstart table without adding a scalar initializer.

For every lookup row r: top=1-2r and hstart=top-r=1-3r.
"""

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
OUTPUT = HERE / "task349_affine_no_scalar.onnx"


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    model = onnx.load(SOURCE)
    arrays = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    hstart = arrays["hstart_offset_by_mod_i8"].astype(np.int16)
    radius = arrays["hend_offset_by_mod_i8"].astype(np.int16)
    assert np.array_equal(hstart, 1 - 3 * radius)
    assert np.all((2 * radius >= 0) & (2 * radius <= 10))

    kept = [x for x in model.graph.initializer if x.name != "hstart_offset_by_mod_i8"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    nodes = []
    removed_hstart = removed_top = 0
    for node in model.graph.node:
        output = node.output[0] if node.output else ""
        if output == "hstart_offset_i8":
            assert node.op_type == "Gather"
            removed_hstart += 1
            continue
        if output == "top_offset_i8":
            assert node.op_type == "Add"
            removed_top += 1
            continue
        nodes.append(node)
        if output == "hend_offset_i8":
            nodes.extend(
                [
                    helper.make_node(
                        "Add",
                        ["hend_offset_i8", "hend_offset_i8"],
                        ["radius_twice_i8"],
                        name="radius_twice_for_offsets",
                    ),
                    helper.make_node(
                        "Sub",
                        ["one_i8", "radius_twice_i8"],
                        ["top_offset_i8"],
                        name="top_offset_from_radius",
                    ),
                    helper.make_node(
                        "Sub",
                        ["top_offset_i8", "hend_offset_i8"],
                        ["hstart_offset_i8"],
                        name="hstart_offset_from_radius",
                    ),
                ]
            )
    assert removed_hstart == 1 and removed_top == 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)
    result = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "identity": "top=1-2r; hstart=top-r=1-3r on all 11 rows",
        "source_profile": profile(SOURCE),
        "candidate_profile": profile(OUTPUT),
    }
    (HERE / "build_no_scalar.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
