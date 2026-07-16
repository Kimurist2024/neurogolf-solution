"""Fail-closed raw-equivalence audit for the exact task245 Selu shave."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = Path("/tmp/root_task245_196/task245.onnx")
CANDIDATE = HERE / "task245_selu_cost384.onnx"
SHARED = ROOT / "scripts/golf/loop_8004_42_plus20/root_selu_scan_127/audit_candidates.py"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH_SEEDS = (245_196_001, 245_196_002)
FRESH_PER_SEED = 5_000


def load_shared():
    spec = importlib.util.spec_from_file_location("selu_audit_shared_196", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load shared audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def operator_model(op: str, count: int) -> bytes:
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [count])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [count])]
    if op == "Div":
        initializers = [
            numpy_helper.from_array(np.asarray(2.0, dtype=np.float16), "two")
        ]
        nodes = [helper.make_node("Div", ["x", "two"], ["y"])]
    elif op == "Selu":
        initializers = []
        nodes = [helper.make_node("Selu", ["x"], ["y"], alpha=1.0, gamma=0.5)]
    else:
        raise ValueError(op)
    graph = helper.make_graph(nodes, f"task245_{op}", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def operator_exhaustive() -> dict[str, object]:
    bits = np.arange(0x0000, 0x7C00, dtype=np.uint16)
    values = bits.view(np.float16)
    rows = {}
    selu_outputs = {}
    for disable, _threads, label in CONFIGS[::2]:
        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if disable
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 4
        outputs = {}
        for op in ("Div", "Selu"):
            session = ort.InferenceSession(
                operator_model(op, len(values)), options, providers=["CPUExecutionProvider"]
            )
            outputs[op] = np.asarray(session.run(None, {"x": values})[0]).view(np.uint16)
        different = np.flatnonzero(outputs["Div"] != outputs["Selu"])
        rows[label] = {
            "different_count": int(different.size),
            "bitwise_equal": not different.size,
        }
        selu_outputs[label] = outputs["Selu"]
    return {
        "input_count": int(values.size),
        "modes": rows,
        "selu_modes_bitwise_equal": bool(
            np.array_equal(
                selu_outputs["disable_all_threads1"],
                selu_outputs["default_threads1"],
            )
        ),
        "pass": bool(all(row["bitwise_equal"] for row in rows.values())),
    }


def structure(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    result: dict[str, object] = {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "params": int(sum(np.asarray(numpy_helper.to_array(item)).size for item in model.graph.initializer)),
        "op_histogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result["full_check"] = False
        result["full_check_error"] = f"{type(exc).__name__}: {exc}"
    for data_prop in (False, True):
        key = "strict_data_prop" if data_prop else "strict"
        try:
            onnx.shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=data_prop
            )
            result[key] = True
        except Exception as exc:  # noqa: BLE001
            result[key] = False
            result[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    ort.set_default_logger_severity(4)
    shared = load_shared()
    authority = AUTHORITY.read_bytes()
    candidate = CANDIDATE.read_bytes()
    report: dict[str, object] = {
        "authority_structure": structure(AUTHORITY),
        "candidate_structure": structure(CANDIDATE),
        "operator_exhaustive_nonnegative_f16": operator_exhaustive(),
        "known_four_configs": {},
        "fresh": [],
    }
    known = shared.known(245)
    for disable, threads, label in CONFIGS:
        report["known_four_configs"][label] = shared.evaluate_cases(
            authority, candidate, known, disable, threads
        )
    for seed in FRESH_SEEDS:
        cases, attempts = shared.generate(245, seed, FRESH_PER_SEED)
        stream = {"seed": seed, "attempts": attempts, "modes": {}}
        for disable, threads, label in CONFIGS:
            stream["modes"][label] = shared.evaluate_cases(
                authority, candidate, cases, disable, threads
            )
        report["fresh"].append(stream)
        print("fresh", seed, len(cases), flush=True)
    comparisons = list(report["known_four_configs"].values()) + [
        mode
        for stream in report["fresh"]
        for mode in stream["modes"].values()
    ]
    report["all_raw_equivalent"] = all(row.get("exact_equivalent") for row in comparisons)
    report["all_truth_correct"] = all(row.get("perfect_truth") for row in comparisons)
    report["runtime_errors_total"] = sum(row.get("runtime_errors_total", 0) for row in comparisons)
    report["candidate_nonfinite_total"] = sum(
        row.get("nonfinite_values", {}).get("candidate", 0) for row in comparisons
    )
    report["pass"] = bool(
        report["operator_exhaustive_nonnegative_f16"]["pass"]
        and report["all_raw_equivalent"]
        and report["all_truth_correct"]
        and report["runtime_errors_total"] == 0
        and report["candidate_nonfinite_total"] == 0
        and report["candidate_structure"]["full_check"]
        and report["candidate_structure"]["strict"]
    )
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS", report["pass"])


if __name__ == "__main__":
    main()
