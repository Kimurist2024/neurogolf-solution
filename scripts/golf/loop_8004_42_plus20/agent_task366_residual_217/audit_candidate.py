#!/usr/bin/env python3
"""Fail-closed task366 audit for the exact lowbit carrier candidate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "others/71407/task366.onnx"
CANDIDATE = HERE / "candidates/task366_lowbit_no_oob.onnx"
SHARED = ROOT / "scripts/golf/loop_8004_42_plus20/root_selu_scan_127/audit_candidates.py"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH = (
    (217_366_001, 1_500),
    (217_366_002, 1_500),
    (217_366_003, 2_500),
    (217_366_004, 2_500),
)

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def load_shared():
    spec = importlib.util.spec_from_file_location("task366_shared_217", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared audit helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_cost(path: Path) -> dict[str, int]:
    model = scoring.sanitize_model(onnx.load(path))
    if model is None:
        raise RuntimeError(f"sanitize failed: {path}")
    with tempfile.TemporaryDirectory(prefix="task366_217_") as directory:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.profile_file_prefix = os.path.join(directory, f"p_{uuid.uuid4().hex}")
        options.log_severity_level = 4
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        zero = np.zeros((1, 10, 30, 30), dtype=np.float32)
        try:
            session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: zero})
        except Exception:
            benchmark = scoring.convert_to_numpy(load_shared().known(366)[0])
            if benchmark is None:
                raise
            session.run(
                [session.get_outputs()[0].name],
                {session.get_inputs()[0].name: benchmark["input"]},
            )
        trace = session.end_profiling()
        memory, params = scoring.score_network(model, trace)
    if memory is None or params is None:
        raise RuntimeError("official scorer returned missing cost")
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def structural(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    row: dict[str, object] = {
        "full_checker": False,
        "strict_shape_inference": False,
        "strict_data_prop_shape_inference": False,
        "static": False,
        "banned_ops": [],
        "nested_graphs": 0,
        "errors": [],
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_checker"] = True
    except Exception as exc:  # noqa: BLE001
        row["errors"].append(f"checker:{type(exc).__name__}:{exc}")
    inferred = None
    try:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True)
        row["strict_shape_inference"] = True
    except Exception as exc:  # noqa: BLE001
        row["errors"].append(f"strict:{type(exc).__name__}:{exc}")
    try:
        shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["strict_data_prop_shape_inference"] = True
    except Exception as exc:  # noqa: BLE001
        row["errors"].append(f"data_prop:{type(exc).__name__}:{exc}")
    banned = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
    row["banned_ops"] = sorted(
        {node.op_type for node in model.graph.node if node.op_type in banned or "Sequence" in node.op_type}
    )
    row["nested_graphs"] = sum(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    if inferred is not None:
        values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
        row["static"] = all(
            value.type.HasField("tensor_type")
            and value.type.tensor_type.HasField("shape")
            and all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in value.type.tensor_type.shape.dim)
            for value in values
        )
    row["pass"] = bool(
        row["full_checker"]
        and row["strict_shape_inference"]
        and row["strict_data_prop_shape_inference"]
        and row["static"]
        and not row["banned_ops"]
        and row["nested_graphs"] == 0
    )
    return row


def main() -> None:
    ort.set_default_logger_severity(4)
    shared = load_shared()
    base_bytes = BASE.read_bytes()
    candidate_bytes = CANDIDATE.read_bytes()
    costs = {"baseline": official_cost(BASE), "candidate": official_cost(CANDIDATE)}
    report: dict[str, object] = {
        "baseline": {"path": str(BASE.relative_to(ROOT)), "sha256": digest(BASE), **costs["baseline"]},
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": digest(CANDIDATE),
            **costs["candidate"],
        },
        "strict_lower": costs["candidate"]["cost"] < costs["baseline"]["cost"],
        "incremental_gain": math.log(costs["baseline"]["cost"] / costs["candidate"]["cost"]),
        "structure": structural(CANDIDATE),
        "runtime_shape_truth": shared.runtime_shape_truth(366, candidate_bytes),
        "known_four_configs": {},
        "fresh": [],
    }
    known_cases = shared.known(366)
    for disable, threads, label in CONFIGS:
        report["known_four_configs"][label] = shared.evaluate_cases(
            base_bytes, candidate_bytes, known_cases, disable, threads
        )
    for seed, count in FRESH:
        cases, attempts = shared.generate(366, seed, count)
        stream = {"seed": seed, "count": len(cases), "attempts": attempts, "configs": {}}
        for disable, threads, label in CONFIGS:
            stream["configs"][label] = shared.evaluate_cases(
                base_bytes, candidate_bytes, cases, disable, threads
            )
        report["fresh"].append(stream)
        print(f"fresh seed={seed} count={len(cases)}", flush=True)
    runtime_rows = list(report["known_four_configs"].values()) + [
        row
        for stream in report["fresh"]
        for row in stream["configs"].values()
    ]
    report["summary"] = {
        "all_raw_equal": all(
            row.get("raw_equal") == row.get("valid")
            - max(row.get("runtime_errors", {}).get("candidate", 0), row.get("runtime_errors", {}).get("baseline", 0))
            for row in runtime_rows
        ),
        "runtime_errors_candidate": sum(row.get("runtime_errors", {}).get("candidate", 0) for row in runtime_rows),
        "runtime_errors_baseline": sum(row.get("runtime_errors", {}).get("baseline", 0) for row in runtime_rows),
        "nonfinite_candidate": sum(row.get("nonfinite_values", {}).get("candidate", 0) for row in runtime_rows),
        "nonfinite_baseline": sum(row.get("nonfinite_values", {}).get("baseline", 0) for row in runtime_rows),
        "minimum_candidate_accuracy": min(
            row.get("candidate_accuracy") or 0.0 for row in runtime_rows
        ),
        "runtime_shape_truthful": bool(report["runtime_shape_truth"].get("truthful", False)),
    }
    (HERE / "audit_no_oob.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
