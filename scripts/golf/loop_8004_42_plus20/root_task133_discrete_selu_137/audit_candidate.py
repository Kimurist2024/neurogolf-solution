#!/usr/bin/env python3
"""Fail-closed raw-equivalence audit for task133 discrete Selu rewrite."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = Path("/private/tmp/ng800946_rank/task133.onnx")
CANDIDATE = HERE / "candidate/task133.onnx"
SELU_LANE = ROOT / "scripts/golf/loop_8004_42_plus20/root_selu_scan_127"
sys.path.insert(0, str(SELU_LANE))
from audit_candidates import evaluate_cases, generate, known, runtime_shape_truth  # noqa: E402


def exhaustive_reachable() -> dict[str, object]:
    values = np.array([-1, 0, 1, 2, 3], dtype=np.float16)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [len(values)])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [len(values)])
    half = numpy_helper.from_array(np.array(0.5, dtype=np.float16), "half")
    mul = helper.make_model(
        helper.make_graph([helper.make_node("Mul", ["x", "half"], ["y"])], "mul", [x], [y], [half]),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    selu = helper.make_model(
        helper.make_graph([
            helper.make_node(
                "Selu", ["x"], ["y"],
                alpha=1.0 / (1.0 - math.exp(-1.0)), gamma=0.5,
            )
        ], "selu", [x], [y]),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    mul.ir_version = selu.ir_version = 9
    rows = {}
    for disable, label in ((True, "disable_all"), (False, "default")):
        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if disable else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        outputs = []
        for model in (mul, selu):
            session = ort.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            outputs.append(np.asarray(session.run(None, {"x": values})[0]))
        rows[label] = {
            "mul_bits": outputs[0].view(np.uint16).astype(int).tolist(),
            "selu_bits": outputs[1].view(np.uint16).astype(int).tolist(),
            "bitwise_equal": bool(np.array_equal(outputs[0].view(np.uint16), outputs[1].view(np.uint16))),
        }
    return {"values": values.astype(float).tolist(), "modes": rows, "pass": all(v["bitwise_equal"] for v in rows.values())}


def main() -> int:
    baseline = AUTHORITY.read_bytes()
    candidate = CANDIDATE.read_bytes()
    known_cases = known(133)
    known_rows = {}
    for disable, threads, label in (
        (True, 1, "disable_all_threads1"),
        (True, 4, "disable_all_threads4"),
        (False, 1, "default_threads1"),
        (False, 4, "default_threads4"),
    ):
        print(f"known {label}", flush=True)
        known_rows[label] = evaluate_cases(baseline, candidate, known_cases, disable, threads)
    fresh_rows = {}
    for seed in (137_133_001, 137_133_002):
        cases, attempts = generate(133, seed, 2500)
        for disable, label in ((True, "disable_all"), (False, "default")):
            key = f"seed{seed}_{label}"
            print(f"fresh {key}", flush=True)
            fresh_rows[key] = evaluate_cases(baseline, candidate, cases, disable, 1)
            fresh_rows[key]["attempts"] = attempts
    exhaustive = exhaustive_reachable()
    rows = list(known_rows.values()) + list(fresh_rows.values())
    result = {
        "task": 133,
        "exhaustive_reachable": exhaustive,
        "known": known_rows,
        "fresh": fresh_rows,
        "authority_runtime_shape": runtime_shape_truth(133, baseline),
        "candidate_runtime_shape": runtime_shape_truth(133, candidate),
        "pass": bool(
            exhaustive["pass"]
            and all(
                row.get("exact_equivalent")
                and row.get("runtime_errors_total") == 0
                and row.get("candidate_accuracy", 0.0) >= 0.90
                for row in rows
            )
        ),
    }
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": result["pass"],
        "reachable_bitwise": exhaustive["pass"],
        "known_min_accuracy": min(row["candidate_accuracy"] for row in known_rows.values()),
        "fresh_min_accuracy": min(row["candidate_accuracy"] for row in fresh_rows.values()),
        "raw_equal": all(row.get("exact_equivalent") for row in rows),
    }, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
