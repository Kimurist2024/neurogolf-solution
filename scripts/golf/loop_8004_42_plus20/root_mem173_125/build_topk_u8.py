#!/usr/bin/env python3
"""Exact task173 transform: rank integral uint8 scores directly with TopK."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "task173.onnx"
OUT = HERE / "task173_topk_u8.onnx"
EXPECTED = "a23d2448c52fe24e949b7758aa754feddfb93012b430fbb1dec10c3e5ce183bf"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    source = SOURCE.read_bytes()
    if sha(source) != EXPECTED:
        raise RuntimeError(f"authority changed: {sha(source)}")
    model = onnx.load_from_string(source)
    aliases = {
        "grid_score": "grid_flat",
        "pix_color": "__ch_h_12",
        "st_center_score": "st_center_score_u8_hx",
        "st_outer_color_pre": "st_center_vals",
        "st_anchor_score": "st_anchor_score_u8_hx",
        "st_anchor_color": "st_anchor_vals",
    }
    removed = {
        "__oh_shape_grid_score", "__oh_src_grid_score", "grid_score", "pix_color",
        "st_center_score", "st_outer_color_pre",
        "st_anchor_score", "st_anchor_color",
    }

    def resolve(name: str) -> str:
        seen = set()
        while name in aliases:
            if name in seen:
                raise RuntimeError(f"alias cycle at {name}")
            seen.add(name)
            name = aliases[name]
        return name

    kept = []
    for node in model.graph.node:
        if any(output in removed for output in node.output):
            continue
        item = copy.deepcopy(node)
        for index, name in enumerate(item.input):
            item.input[index] = resolve(name)
        kept.append(item)
    del model.graph.node[:]
    model.graph.node.extend(kept)

    for value in list(model.graph.value_info) + list(model.graph.output):
        if value.name in {
            "__oh_pre_pix_vals", "__ch_h_12", "st_center_score_u8_hx",
            "st_center_vals", "st_anchor_score_u8_hx", "st_anchor_vals",
        }:
            value.type.tensor_type.elem_type = TensorProto.UINT8
    kept_vi = [value for value in model.graph.value_info if value.name not in removed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUT)
    result = {
        "authority_sha256": sha(source),
        "candidate_sha256": sha(OUT.read_bytes()),
        "removed_nodes": len(removed),
        "source_nodes": len(onnx.load_from_string(source).graph.node),
        "candidate_nodes": len(model.graph.node),
        "proof": "TopK order/ties are unchanged by removing exact uint8-to-float16 casts for values in [0,9]",
    }
    (HERE / "topk_u8_build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
