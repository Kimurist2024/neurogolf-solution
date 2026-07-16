#!/usr/bin/env python3
"""Conservative support scan for removing the fp16 log epsilon."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent


def main() -> None:
    values: set[float] = set()
    weights = np.asarray([2.0 ** (15 - col) for col in range(19)], dtype=np.float16)
    # Original rows span <=5 bits and magnified rows span <=10 bits.  Enumerate
    # every non-empty mask in every contiguous window up to that conservative
    # maximum, a superset of generator-reachable row masks.
    for width in range(1, 11):
        for start in range(20 - width):
            for mask in range(1, 1 << width):
                value = np.float16(sum(
                    float(weights[start + offset])
                    for offset in range(width) if (mask >> offset) & 1
                ))
                values.add(float(value))
    inputs = np.asarray(sorted(values), dtype=np.float16)
    count = len(inputs)
    nodes = []
    inits = []
    outputs = []
    variants = {
        "current": (np.float16(1.4423828125), np.float16(0.005001068115234375)),
        "low_no_epsilon": (np.float16(1.4423828125), np.float16(0.0)),
        "high_no_epsilon": (np.float16(1.443359375), np.float16(0.0)),
    }
    for label, (inverse, epsilon) in variants.items():
        inits.extend([
            numpy_helper.from_array(np.asarray(inverse, np.float16), f"{label}_inverse"),
            numpy_helper.from_array(np.asarray(epsilon, np.float16), f"{label}_epsilon"),
        ])
        nodes.extend([
            helper.make_node("Log", ["input"], [f"{label}_log"]),
            helper.make_node("Mul", [f"{label}_log", f"{label}_inverse"], [f"{label}_mul"]),
            helper.make_node("Add", [f"{label}_mul", f"{label}_epsilon"], [f"{label}_add"]),
            helper.make_node("Floor", [f"{label}_add"], [label]),
        ])
        outputs.append(helper.make_tensor_value_info(label, TensorProto.FLOAT16, [count]))
    model = helper.make_model(
        helper.make_graph(
            nodes, "task319_scale_support",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [count])],
            outputs, inits,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    model.ir_version = 8
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    current, low, high = session.run(None, {"input": inputs})
    result = {"support_values": count, "min": float(inputs.min()), "max": float(inputs.max())}
    for label, output in (("low_no_epsilon", low), ("high_no_epsilon", high)):
        indices = np.flatnonzero(output != current)
        result[label] = {
            "differences": int(indices.size),
            "examples": [
                {"input": float(inputs[i]), "current": float(current[i]), "candidate": float(output[i])}
                for i in indices[:20]
            ],
        }
    (HERE / "scale_support_scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
