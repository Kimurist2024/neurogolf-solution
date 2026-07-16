#!/usr/bin/env python3
"""Exhaust Add(x,-1) versus Sub(x,+1) over every float16 bit pattern."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
BITS = np.arange(1 << 16, dtype=np.uint16)
VALUES = BITS.view(np.float16)
CONFIGS = (
    (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1, "disable_all_threads1"),
    (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4, "disable_all_threads4"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1, "default_threads1"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4, "default_threads4"),
)


def make_model(op: str, constant: float) -> bytes:
    graph = helper.make_graph(
        [helper.make_node(op, ["input", "constant"], ["output"])],
        f"{op}_one",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [VALUES.size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [VALUES.size])],
        [numpy_helper.from_array(np.asarray(constant, dtype=np.float16), name="constant")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def run(data: bytes, level: ort.GraphOptimizationLevel, threads: int) -> np.ndarray:
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = options.inter_op_num_threads = threads
    options.log_severity_level = 4
    session = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
    return np.asarray(session.run(["output"], {"input": VALUES})[0])


def main() -> None:
    add = make_model("Add", -1.0)
    sub = make_model("Sub", 1.0)
    rows = []
    for level, threads, label in CONFIGS:
        expected = run(add, level, threads).view(np.uint16)
        actual = run(sub, level, threads).view(np.uint16)
        unequal = expected != actual
        rows.append(
            {
                "label": label,
                "raw_equal": bool(not np.any(unequal)),
                "difference_count": int(np.count_nonzero(unequal)),
                "first_difference_bits": int(BITS[np.flatnonzero(unequal)[0]]) if np.any(unequal) else None,
            }
        )
    payload = {
        "input_patterns": int(BITS.size),
        "rows": rows,
        "pass": all(row["raw_equal"] for row in rows),
    }
    (HERE / "exhaust_add_sub_one.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
