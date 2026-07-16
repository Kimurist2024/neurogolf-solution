#!/usr/bin/env python3
"""Execute all 80 changed factor entries under the four ORT configurations."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep33/shared_sign/task333_r01.onnx"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item).astype(np.float32) for item in model.graph.initializer}


def diagnostic(values: dict[str, np.ndarray], candidate: bool) -> bytes:
    initializers = []
    for name in (("HC", "GHHT") if candidate else ("HC", "GHHT", "GE")):
        initializers.append(numpy_helper.from_array(values[name], name=name))
    if candidate:
        first = helper.make_node("Identity", ["HC"], ["first"])
    else:
        first = helper.make_node("Einsum", ["HC", "GE"], ["first"], equation="Zd,Z->Zd")
    second = helper.make_node("Einsum", ["GHHT", "HC"], ["second"], equation="tU,Uc->tUc")
    graph = helper.make_graph(
        [first, second],
        "task333_changed_factor_support",
        [],
        [
            helper.make_tensor_value_info("first", TensorProto.FLOAT, [2, 10]),
            helper.make_tensor_value_info("second", TensorProto.FLOAT, [3, 2, 10]),
        ],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def session(data: bytes, disable: bool, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])


def main() -> None:
    with zipfile.ZipFile(ROOT / "submission_base_8005.17.zip") as archive:
        base = onnx.load_from_string(archive.read("task333.onnx"))
    candidate = onnx.load(CANDIDATE)
    base_graph = diagnostic(arrays(base), False)
    candidate_graph = diagnostic(arrays(candidate), True)
    rows = {}
    for disable, threads, label in CONFIGS:
        try:
            old = session(base_graph, disable, threads).run(None, {})
            new = session(candidate_graph, disable, threads).run(None, {})
            exact = all(np.array_equal(left, right) for left, right in zip(old, new))
            nonfinite = sum(int(array.size - np.count_nonzero(np.isfinite(array))) for array in old + new)
            rows[label] = {
                "runtime_errors": 0,
                "entries": sum(array.size for array in old),
                "different_entries": sum(int(np.count_nonzero(left != right)) for left, right in zip(old, new)),
                "nonfinite_values": nonfinite,
                "max_abs_difference": max(float(np.abs(left - right).max(initial=0.0)) for left, right in zip(old, new)),
                "exact": exact and nonfinite == 0,
            }
        except Exception as exc:  # noqa: BLE001
            rows[label] = {"runtime_errors": 1, "error": f"{type(exc).__name__}: {exc}", "exact": False}
    result = {
        "task": 333,
        "support": "20 first-use (Z,d) entries + 60 shared-use (t,U,c) entries",
        "expected_entries": 80,
        "configs": rows,
        "complete": all(row.get("entries") == 80 and row.get("exact") for row in rows.values()),
    }
    (HERE / "factor_support_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
