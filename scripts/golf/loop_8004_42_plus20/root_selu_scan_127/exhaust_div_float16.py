#!/usr/bin/env python3
"""Exhaustively prove ORT float16 Div(x,c) == Selu(x,gamma=1/c), x>=0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
DIVISORS = {
    "task090_ln2": np.float16(0.693),
    "task209_ln2": np.float16(0.6934),
}


def model(op: str, divisor: np.float16, count: int) -> bytes:
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [count])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [count])]
    if op == "Div":
        initializers = [numpy_helper.from_array(np.asarray(divisor, dtype=np.float16), "c")]
        nodes = [helper.make_node("Div", ["x", "c"], ["y"], name="div")]
    elif op == "Selu":
        initializers = []
        nodes = [
            helper.make_node(
                "Selu",
                ["x"],
                ["y"],
                name="selu",
                alpha=1.0,
                gamma=1.0 / float(divisor),
            )
        ]
    else:
        raise ValueError(op)
    graph = helper.make_graph(nodes, f"exhaust_{op}", inputs, outputs, initializer=initializers)
    value = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10)
    onnx.checker.check_model(value, full_check=True)
    return value.SerializeToString()


def session(data: bytes, disable: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])


def main() -> int:
    bits = np.arange(0x0000, 0x7C00, dtype=np.uint16)
    values = bits.view(np.float16)
    report = {
        "domain": "all nonnegative finite float16 bit patterns",
        "input_count": int(values.size),
        "rows": {},
    }
    for label, divisor in DIVISORS.items():
        row = {
            "divisor_float16": float(divisor),
            "gamma_float32": float(np.float32(1.0 / float(divisor))),
            "modes": {},
        }
        mode_outputs = {}
        for disable, mode in ((True, "disable_all"), (False, "default")):
            div = session(model("Div", divisor, values.size), disable).run(["y"], {"x": values})[0]
            selu = session(model("Selu", divisor, values.size), disable).run(["y"], {"x": values})[0]
            div_bits = np.asarray(div, dtype=np.float16).view(np.uint16)
            selu_bits = np.asarray(selu, dtype=np.float16).view(np.uint16)
            different = np.flatnonzero(div_bits != selu_bits)
            row["modes"][mode] = {
                "bitwise_equal": not different.size,
                "different_count": int(different.size),
                "first_differences": [
                    {
                        "input_bits": int(bits[index]),
                        "input": float(values[index]),
                        "div_bits": int(div_bits[index]),
                        "selu_bits": int(selu_bits[index]),
                    }
                    for index in different[:20]
                ],
            }
            mode_outputs[mode] = selu_bits
        row["selu_modes_bitwise_equal"] = bool(
            np.array_equal(mode_outputs["disable_all"], mode_outputs["default"])
        )
        row["pass"] = bool(
            all(item["bitwise_equal"] for item in row["modes"].values())
            and row["selu_modes_bitwise_equal"]
        )
        report["rows"][label] = row
        print(label, row["pass"], flush=True)
    report["pass"] = all(row["pass"] for row in report["rows"].values())
    (HERE / "exhaust_div_float16.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
