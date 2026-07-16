#!/usr/bin/env python3
"""Build truthful two-Einsum task243 variants with additional flood steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE / "candidates/task243_truthful_constant.onnx"
OUTPUT = HERE / "chunked_build.json"
EXTRA_STEPS = (8, 12, 16, 20)
FREE_LABELS = "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZbcd"


def get_equation(node: onnx.NodeProto) -> str:
    return next(
        helper.get_attribute_value(attr).decode("ascii")
        for attr in node.attribute
        if attr.name == "equation"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(extra_steps: int) -> Path:
    base = onnx.load(BASE)
    original = base.graph.node[0]
    terms, output = get_equation(original).split("->", 1)
    terms = terms.split(",")
    inputs = list(original.input)
    if output != "afVU" or terms[-3:] != ["ef", "fe", "f"]:
        raise RuntimeError("unexpected truthful base terminal suffix")

    walk = helper.make_node(
        "Einsum",
        inputs[:-3],
        ["reach"],
        name="truthful_walk_chunk0",
        equation=",".join(terms[:-3]) + "->aeVU",
    )

    first, second = "g", "h"
    continuation_terms = ["aegh"]
    continuation_inputs = ["reach"]
    for index, new_label in enumerate(FREE_LABELS[:extra_steps]):
        if index % 2 == 0:
            continuation_terms.extend(
                [second + new_label, new_label + second, "ae" + first + new_label]
            )
            second = new_label
        else:
            continuation_terms.extend(
                [first + new_label, new_label + first, "ae" + new_label + second]
            )
            first = new_label
        continuation_inputs.extend(["L", "L", "input"])
    continuation_terms.extend(["ef", "fe", "f"])
    continuation_inputs.extend(["CB", "CB", "w_dyn"])
    finish = helper.make_node(
        "Einsum",
        continuation_inputs,
        ["output"],
        name=f"truthful_walk_chunk1_plus{extra_steps}",
        equation=",".join(continuation_terms) + "->af" + first + second,
    )

    del base.graph.node[:]
    base.graph.node.extend([walk, finish])
    del base.graph.value_info[:]
    base.graph.value_info.extend(
        [
            helper.make_tensor_value_info(
                "reach", TensorProto.FLOAT, [1, 10, 30, 30]
            )
        ]
    )
    base.producer_name = "codex-task243-truthful-chunked-repair"
    onnx.checker.check_model(base, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        base, strict_mode=True, data_prop=True
    )
    shapes = {
        value.name: [
            int(dim.dim_value) for dim in value.type.tensor_type.shape.dim
        ]
        for value in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
    }
    if shapes.get("reach") != [1, 10, 30, 30] or shapes.get("output") != [
        1,
        10,
        30,
        30,
    ]:
        raise RuntimeError(f"plus{extra_steps}: non-truthful shapes {shapes}")
    path = HERE / f"candidates/task243_truthful_chunked_plus{extra_steps}.onnx"
    onnx.save(base, path)
    return path


def main() -> None:
    rows = []
    for steps in EXTRA_STEPS:
        path = build(steps)
        model = onnx.load(path)
        rows.append(
            {
                "extra_steps": steps,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "einsum_inputs": [len(node.input) for node in model.graph.node],
                "equations": [get_equation(node) for node in model.graph.node],
                "truthful_intermediate": {"reach": [1, 10, 30, 30]},
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
