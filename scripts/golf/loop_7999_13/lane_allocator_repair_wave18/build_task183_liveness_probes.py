#!/usr/bin/env python3
"""Build cheap liveness anchors for task183 after removing its dead variadic Min."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


SOURCES = [
    "tl_u8",
    "tr5_u8",
    "tr7_u8",
    "tr9_u8",
    "tr_u8",
    "bl5_u8",
    "bl7_u8",
    "bl9_u8",
    "bl_u8",
    "br5_u8",
    "br7_u8",
    "br9_u8",
    "br_u8",
]


def insert_after_quad6(model: onnx.ModelProto, nodes: list[onnx.NodeProto]) -> None:
    index = next(
        index
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Resize" and "quad6" in node.output
    )
    original = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(original[: index + 1] + nodes + original[index + 1 :])


def save_checked(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = onnx.load(args.input)
    rows: list[dict[str, object]] = []

    for ordinal, tensor in enumerate(SOURCES, 1):
        for op_type in ("Identity", "Shape", "Size", "ReduceMax"):
            model = copy.deepcopy(source)
            output = f"allocator_anchor_{op_type.lower()}_{ordinal:02d}"
            attributes: dict[str, int] = {}
            if op_type == "ReduceMax":
                attributes["keepdims"] = 0
            node = helper.make_node(
                op_type,
                [tensor],
                [output],
                name=f"allocator_anchor_{op_type.lower()}_{ordinal:02d}",
                **attributes,
            )
            insert_after_quad6(model, [node])
            path = args.output_dir / f"task183_{op_type.lower()}_{ordinal:02d}.onnx"
            save_checked(model, path)
            rows.append({"path": str(path), "op_type": op_type, "sources": [tensor]})

    # Also test one scalar reduction per source. This retains every former Min
    # input without materializing the large elementwise Min output.
    model = copy.deepcopy(source)
    nodes = [
        helper.make_node(
            "ReduceMax",
            [tensor],
            [f"allocator_anchor_all_{ordinal:02d}"],
            name=f"allocator_anchor_all_{ordinal:02d}",
            keepdims=0,
        )
        for ordinal, tensor in enumerate(SOURCES, 1)
    ]
    insert_after_quad6(model, nodes)
    path = args.output_dir / "task183_reducemax_all.onnx"
    save_checked(model, path)
    rows.append({"path": str(path), "op_type": "ReduceMax", "sources": SOURCES})

    manifest = args.output_dir / "build_manifest.json"
    manifest.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
