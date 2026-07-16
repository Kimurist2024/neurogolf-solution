#!/usr/bin/env python3
"""Exhaust the task366 log2 Round-removal carrier over all reachable lowbits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
VALUES = np.asarray([0] + [1 << shift for shift in range(32)], dtype=np.uint32)
CONFIGS = (
    (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1, "disable_all_threads1"),
    (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4, "disable_all_threads4"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1, "default_threads1"),
    (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4, "default_threads4"),
)


def make_model(round_value: bool) -> bytes:
    nodes = [
        helper.make_node("Cast", ["input"], ["f16"], to=TensorProto.FLOAT16),
        helper.make_node("Log", ["f16"], ["logged"]),
        helper.make_node("Div", ["logged", "ln2"], ["quotient"]),
    ]
    source = "quotient"
    if round_value:
        nodes.append(helper.make_node("Round", [source], ["rounded"]))
        source = "rounded"
    nodes.append(helper.make_node("Cast", [source], ["output"], to=TensorProto.INT32))
    graph = helper.make_graph(
        nodes,
        "round_carrier",
        [helper.make_tensor_value_info("input", TensorProto.UINT32, [VALUES.size])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [VALUES.size])],
        [numpy_helper.from_array(np.asarray(0.693359375, dtype=np.float16), name="ln2")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def make_search_model(rows: int) -> bytes:
    nodes = [
        helper.make_node("Cast", ["input"], ["f16"], to=TensorProto.FLOAT16),
        helper.make_node("Log", ["f16"], ["logged"]),
        helper.make_node("Div", ["logged", "divisor"], ["quotient"]),
        helper.make_node("Cast", ["quotient"], ["output"], to=TensorProto.INT32),
    ]
    graph = helper.make_graph(
        nodes,
        "divisor_search",
        [
            helper.make_tensor_value_info("input", TensorProto.UINT32, [rows, VALUES.size]),
            helper.make_tensor_value_info("divisor", TensorProto.FLOAT16, [rows, 1]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [rows, VALUES.size])],
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
    baseline = make_model(True)
    candidate = make_model(False)
    rows = []
    for level, threads, label in CONFIGS:
        expected = run(baseline, level, threads)
        actual = run(candidate, level, threads)
        rows.append(
            {
                "label": label,
                "values": VALUES.astype(int).tolist(),
                "baseline": expected.tolist(),
                "candidate": actual.tolist(),
                "equal": bool(np.array_equal(expected, actual)),
                "differences": int(np.count_nonzero(expected != actual)),
            }
        )
    raw_bits = np.arange(0x3800, 0x3A00, dtype=np.uint16)
    divisors = raw_bits.view(np.float16)
    matrix = np.broadcast_to(VALUES, (divisors.size, VALUES.size)).copy()
    search_rows = []
    search_model = make_search_model(divisors.size)
    common: set[int] | None = None
    for level, threads, label in CONFIGS:
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = options.inter_op_num_threads = threads
        options.log_severity_level = 4
        session = ort.InferenceSession(search_model, options, providers=["CPUExecutionProvider"])
        actual = np.asarray(
            session.run(["output"], {"input": matrix, "divisor": divisors[:, None]})[0]
        )
        expected = run(baseline, level, threads)
        expected_matrix = np.broadcast_to(expected, matrix.shape)
        passing = set(int(index) for index in np.flatnonzero(np.all(actual == expected_matrix, axis=1)))
        common = passing if common is None else common & passing
        search_rows.append({"label": label, "passing_count": len(passing)})
    common = common or set()
    common_bits = [int(raw_bits[index]) for index in sorted(common)]
    common_values = [float(divisors[index]) for index in sorted(common)]
    payload = {
        "support": "zero plus every uint32 power of two",
        "count": int(VALUES.size),
        "rows": rows,
        "pass": all(row["equal"] for row in rows),
        "divisor_search": {
            "bit_range": [int(raw_bits[0]), int(raw_bits[-1])],
            "rows": search_rows,
            "common_passing_count": len(common),
            "common_bits": common_bits,
            "common_values": common_values,
        },
    }
    (HERE / "exhaust_round_carrier.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
