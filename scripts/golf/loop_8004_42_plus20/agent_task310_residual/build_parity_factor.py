#!/usr/bin/env python3
"""Factor task310's 4-way even-parity A2 tensor exactly."""

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[4]
SOURCE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_mask_absorb_174/base/task310.onnx"
)
OUTPUT = Path(__file__).resolve().parent / "task310_exact_parity_factor.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    final = model.graph.node[-1]
    assert final.op_type == "Einsum"

    a2_tensor = next(t for t in model.graph.initializer if t.name == "A2")
    a2 = numpy_helper.to_array(a2_tensor)
    h = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    weight = np.asarray([0.5, 0.5], dtype=np.float32)
    rebuilt = np.einsum("kd,kr,kj,kc,k->drjc", h, h, h, h, weight)
    assert np.array_equal(a2, rebuilt)

    kept = [t for t in model.graph.initializer if t.name != "A2"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(h, name="ParityH"),
            numpy_helper.from_array(weight, name="ParityW"),
        ]
    )

    equation_attr = next(a for a in final.attribute if a.name == "equation")
    equation = helper.get_attribute_value(equation_attr).decode("ascii")
    lhs, rhs = equation.split("->")
    terms = lhs.split(",")
    inputs = list(final.input)
    assert inputs[8] == "A2" and terms[8] == "Drjc"
    assert inputs[26] == "A2" and terms[26] == "Gltf"

    new_inputs: list[str] = []
    new_terms: list[str] = []
    for index, (name, term) in enumerate(zip(inputs, terms)):
        if index == 8:
            new_inputs.extend(["ParityH"] * 4 + ["ParityW"])
            new_terms.extend(["kD", "kr", "kj", "kc", "k"])
        elif index == 26:
            new_inputs.extend(["ParityH"] * 4 + ["ParityW"])
            new_terms.extend(["zG", "zl", "zt", "zf", "z"])
        else:
            new_inputs.append(name)
            new_terms.append(term)

    del final.input[:]
    final.input.extend(new_inputs)
    equation_attr.s = (",".join(new_terms) + "->" + rhs).encode("ascii")

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)


if __name__ == "__main__":
    main()
