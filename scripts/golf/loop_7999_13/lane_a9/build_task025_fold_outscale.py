#!/usr/bin/env python3
"""Fold task025's repeated scalar outscale into vsgn and existing negsigK."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task025.onnx"
OUTPUT = HERE / "task025_fold_outscale.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    graph = model.graph
    final = graph.node[-1]
    assert final.op_type == "Einsum" and final.output == ["output"]
    old_inputs = list(final.input)
    positions = [index for index, name in enumerate(old_inputs) if name == "outscale"]
    assert len(positions) == 15

    equation_attr = next(attr for attr in final.attribute if attr.name == "equation")
    equation = helper.get_attribute_value(equation_attr).decode("ascii")
    lhs, rhs = equation.split("->")
    terms = lhs.split(",")
    assert len(terms) == len(old_inputs)
    assert all(terms[index] == "ijkl" for index in positions)

    # Fourteen scalar operands become existing +1.  The final one contracts
    # existing negsigK=[0,-240] over a new private z axis, yielding -240.
    # Absorb sqrt(1e300/240) into each of the two existing vsgn factors:
    #   (-240) * (vsgn*f)^2 == (-1e20)^15 * vsgn^2.
    for index in positions[:-1]:
        final.input[index] = "one"
    pivot = positions[-1]
    final.input[pivot] = "negsigK"
    terms[pivot] = "ijzl"
    equation_attr.s = (",".join(terms) + "->" + rhs).encode("ascii")

    init_map = {item.name: item for item in graph.initializer}
    old_vsgn = numpy_helper.to_array(init_map["vsgn"]).astype(np.float64)
    factor = math.sqrt(1.0e300 / 240.0)
    new_vsgn = old_vsgn * factor
    del graph.initializer[:]
    graph.initializer.extend(
        numpy_helper.from_array(new_vsgn, name="vsgn") if item.name == "vsgn" else item
        for item in init_map.values()
        if item.name != "outscale"
    )

    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, OUTPUT)
    data = OUTPUT.read_bytes()
    payload = {
        "source": str(SOURCE.relative_to(HERE.parents[3])),
        "candidate": str(OUTPUT.relative_to(HERE.parents[3])),
        "sha256": hashlib.sha256(data).hexdigest(),
        "removed_initializer": "outscale:float64[1,1,1,1]",
        "parameter_reduction": 1,
        "node_count_unchanged": len(model.graph.node),
        "einsum_input_count_unchanged": len(final.input),
        "fold_factor": factor,
    }
    (HERE / "task025_fold_build.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
