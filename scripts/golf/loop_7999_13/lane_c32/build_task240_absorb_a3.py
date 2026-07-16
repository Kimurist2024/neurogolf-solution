#!/usr/bin/env python3
"""Exactly absorb broadcast A3 into the globally reused B row gauge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task240.onnx"
OUTPUT = HERE / "task240_absorb_a3.onnx"
FACTORS = {"A0", "A2", "A4", "U0", "U1", "U2", "U3", "U4"}


def main() -> None:
    model = onnx.load(SOURCE)
    node = model.graph.node[0]
    equation_attr = next(attr for attr in node.attribute if attr.name == "equation")
    left, output_term = equation_attr.s.decode("ascii").split("->")
    terms = left.split(",")
    if len(terms) != len(node.input):
        raise RuntimeError("equation/input mismatch")
    arrays = {item.name: numpy_helper.to_array(item).copy() for item in model.graph.initializer}
    gauge = arrays["A3"][0].astype(np.float64)
    if np.any(gauge == 0):
        raise RuntimeError("noninvertible A3 gauge")

    # Every B row receives A3. Every retained paired factor receives A3^-1.
    arrays["B"] = (arrays["B"].astype(np.float64) * gauge[:, None]).astype(np.float32)
    for name in FACTORS:
        arrays[name] = (arrays[name].astype(np.float64) / gauge[None, :]).astype(np.float32)

    kept_pairs = [(name, term) for name, term in zip(node.input, terms) if name != "A3"]
    removed = len(node.input) - len(kept_pairs)
    if removed != 2:
        raise RuntimeError(f"expected two A3 operands, got {removed}")
    del node.input[:]
    node.input.extend(name for name, _ in kept_pairs)
    equation_attr.s = (",".join(term for _, term in kept_pairs) + "->" + output_term).encode("ascii")

    kept_names = [item.name for item in model.graph.initializer if item.name != "A3"]
    del model.graph.initializer[:]
    for name in kept_names:
        model.graph.initializer.append(numpy_helper.from_array(arrays[name], name=name))
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT)
    (HERE / "task240_absorb_a3_build.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE),
                "output": str(OUTPUT),
                "gauge": gauge.tolist(),
                "removed_initializer": "A3",
                "removed_params": 2,
                "removed_operands": 2,
                "remaining_operands": len(node.input),
                "transformed_factors": sorted(FACTORS | {"B"}),
                "algebra": "B'=diag(A3)B; F'=F diag(A3)^-1; remove both A3 factors",
            },
            indent=2,
        )
        + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
