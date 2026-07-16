#!/usr/bin/env python3
"""Extend task243's single-Einsum flood walk using all five free labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE / "candidates/task243_truthful_constant.onnx"
OUTPUT = HERE / "extended_build.json"
FREE_LABELS = "WXYZd"


def equation(node: onnx.NodeProto) -> str:
    return next(
        onnx.helper.get_attribute_value(attr).decode("ascii")
        for attr in node.attribute
        if attr.name == "equation"
    )


def set_equation(node: onnx.NodeProto, value: str) -> None:
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = value.encode("ascii")
            return
    raise RuntimeError("Einsum has no equation attribute")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(extra_steps: int) -> Path:
    model = onnx.load(BASE)
    node = model.graph.node[0]
    if node.op_type != "Einsum" or len(model.graph.node) != 1:
        raise RuntimeError("unexpected truthful base topology")
    left, right = equation(node).split("->", 1)
    terms = left.split(",")
    inputs = list(node.input)
    if right != "afVU" or terms[-3:] != ["ef", "fe", "f"]:
        raise RuntimeError("unexpected task243 terminal suffix")

    body_terms, decode_terms = terms[:-3], terms[-3:]
    body_inputs, decode_inputs = inputs[:-3], inputs[-3:]
    first, second = "V", "U"
    added_terms: list[str] = []
    added_inputs: list[str] = []
    for index, new_label in enumerate(FREE_LABELS[:extra_steps]):
        if index % 2 == 0:
            added_terms.extend(
                [second + new_label, new_label + second, "ae" + first + new_label]
            )
            second = new_label
        else:
            added_terms.extend(
                [first + new_label, new_label + first, "ae" + new_label + second]
            )
            first = new_label
        added_inputs.extend(["L", "L", "input"])

    del node.input[:]
    node.input.extend(body_inputs + added_inputs + decode_inputs)
    set_equation(
        node,
        ",".join(body_terms + added_terms + decode_terms)
        + "->af"
        + first
        + second,
    )
    node.name = f"truthful_terminal_einsum_plus{extra_steps}"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    output_shape = [
        int(dim.dim_value)
        for dim in inferred.graph.output[0].type.tensor_type.shape.dim
    ]
    if output_shape != [1, 10, 30, 30]:
        raise RuntimeError(f"plus{extra_steps}: wrong output shape {output_shape}")
    path = HERE / f"candidates/task243_truthful_constant_plus{extra_steps}.onnx"
    onnx.save(model, path)
    return path


def main() -> None:
    rows = []
    for extra_steps in range(1, len(FREE_LABELS) + 1):
        path = build(extra_steps)
        model = onnx.load(path)
        node = model.graph.node[0]
        rows.append(
            {
                "extra_steps": extra_steps,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "einsum_inputs": len(node.input),
                "equation": equation(node),
            }
        )
        print(path)
    OUTPUT.write_text(
        json.dumps({"base": str(BASE.relative_to(ROOT)), "variants": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
