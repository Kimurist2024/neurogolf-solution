#!/usr/bin/env python3
"""Fail-closed truthful-shape/gold/margin/fresh audit for task071."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidates/task071_truthful_dense.onnx"
OUTPUT = HERE / "evidence.json"
AUTHORITY_ZIP = ROOT / "submission.zip"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
FRESH_SEEDS = [71_20260715, 71_20260716]
FRESH_CASES_PER_SEED = 2000

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str | None:
    return digest(path.read_bytes()) if path.is_file() else None


def concrete_shape(value: onnx.ValueInfoProto) -> list[int]:
    tensor_type = value.type.tensor_type
    if not value.type.HasField("tensor_type") or not tensor_type.HasField("shape"):
        raise RuntimeError(f"missing tensor shape/type: {value.name}")
    shape: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value"):
            raise RuntimeError(f"non-static dimension: {value.name}")
        if dim.dim_value <= 0:
            raise RuntimeError(f"non-positive dimension: {value.name}")
        shape.append(int(dim.dim_value))
    return shape


def static_audit(model: onnx.ModelProto) -> tuple[dict[str, Any], onnx.ModelProto]:
    onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    tensors = [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    shapes = {value.name: concrete_shape(value) for value in tensors}
    banned = {
        "LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"
    }
    bad_ops = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in banned or "Sequence" in node.op_type
        }
    )
    nested = sum(
        attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        for node in model.graph.node
        for attr in node.attribute
    )
    finite = all(
        bool(np.all(np.isfinite(onnx.numpy_helper.to_array(init))))
        for init in model.graph.initializer
    )
    if bad_ops or nested or model.functions or not finite:
        raise RuntimeError(
            f"static rejection: banned={bad_ops}, nested={nested}, "
            f"functions={len(model.functions)}, finite={finite}"
        )
    if shapes.get("input") != [1, 10, 30, 30]:
        raise RuntimeError(f"wrong input shape: {shapes.get('input')}")
    if shapes.get("output") != [1, 10, 30, 30]:
        raise RuntimeError(f"wrong output shape: {shapes.get('output')}")
    return (
        {
            "full_check": True,
            "strict_shape_inference_data_prop": True,
            "all_shapes_static_positive": True,
            "shapes": shapes,
            "banned_ops": bad_ops,
            "nested_graphs": nested,
            "functions": len(model.functions),
            "finite_initializers": finite,
        },
        inferred,
    )


def make_session(model: onnx.ModelProto) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected candidate")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def runtime_shape_audit(inferred: onnx.ModelProto) -> dict[str, Any]:
    """Expose every node result and prove its runtime shape matches its declaration."""
    generator = importlib.import_module("task_3345333e")
    random.seed(710_710_071)
    np.random.seed(710_710_071)
    benchmark = scoring.convert_to_numpy(generator.generate())
    if benchmark is None:
        raise RuntimeError("generator emitted oversized example")

    probe = copy.deepcopy(inferred)
    typed = {
        value.name: value
        for value in [*probe.graph.value_info, *probe.graph.output]
    }
    names = [output for node in probe.graph.node for output in node.output if output]
    del probe.graph.output[:]
    for name in names:
        if name not in typed:
            raise RuntimeError(f"inference omitted node output: {name}")
        probe.graph.output.append(copy.deepcopy(typed[name]))
    del probe.graph.value_info[:]
    onnx.checker.check_model(probe, full_check=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        probe.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    values = session.run(None, {"input": benchmark["input"]})
    runtime = {name: list(value.shape) for name, value in zip(names, values)}
    declared = {name: concrete_shape(typed[name]) for name in names}
    mismatches = {
        name: {"declared": declared[name], "runtime": runtime[name]}
        for name in names
        if declared[name] != runtime[name]
    }
    if mismatches:
        raise RuntimeError(f"runtime/declaration shape mismatch: {mismatches}")
    return {
        "all_node_outputs_checked": len(names),
        "declared_equals_runtime": True,
        "mismatches": mismatches,
        "runtime_shapes": runtime,
        "shape_cloak": False,
    }


def profile(model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="task071_truthful_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(model), 71, workdir, label="truthful", require_correct=True
        )
    if result is None:
        raise RuntimeError("official gold/scoring gate rejected candidate")
    return result


def fresh_audit(model: onnx.ModelProto, seed: int) -> dict[str, Any]:
    generator = importlib.import_module("task_3345333e")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    session = make_session(model)
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    right = wrong = runtime_errors = nonfinite = shape_mismatch = 0
    near_boundary_values = 0
    min_nonzero_abs = math.inf
    first_failure = None
    raw_hash = hashlib.sha256()
    started = time.monotonic()

    for index in range(FRESH_CASES_PER_SEED):
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"fresh[{index}] unexpectedly exceeds 30x30")
        try:
            raw = session.run([output_name], {input_name: benchmark["input"]})[0]
        except BaseException as exc:
            runtime_errors += 1
            wrong += 1
            if first_failure is None:
                first_failure = {"index": index, "reason": f"{type(exc).__name__}: {exc}"}
            continue
        raw_hash.update(np.ascontiguousarray(raw).tobytes())
        if raw.shape != benchmark["output"].shape:
            shape_mismatch += 1
        if not np.all(np.isfinite(raw)):
            nonfinite += 1
        abs_raw = np.abs(raw)
        nonzero = abs_raw[abs_raw > 0]
        if nonzero.size:
            min_nonzero_abs = min(min_nonzero_abs, float(nonzero.min()))
            near_boundary_values += int(np.count_nonzero(nonzero < 0.25))
        prediction = (raw > 0).astype(np.float32)
        if np.array_equal(prediction, benchmark["output"]):
            right += 1
        else:
            wrong += 1
            if first_failure is None:
                bad = np.argwhere(prediction != benchmark["output"])
                first_failure = {
                    "index": index,
                    "reason": "gold mismatch",
                    "first_bad_index": bad[0].tolist() if bad.size else None,
                }
        if (index + 1) % 250 == 0:
            print(
                f"seed={seed} fresh {index + 1}/{FRESH_CASES_PER_SEED}: "
                f"right={right} wrong={wrong}",
                flush=True,
            )
    return {
        "seed": seed,
        "cases": FRESH_CASES_PER_SEED,
        "right": right,
        "wrong": wrong,
        "accuracy": right / FRESH_CASES_PER_SEED,
        "runtime_errors": runtime_errors,
        "nonfinite_cases": nonfinite,
        "shape_mismatch_cases": shape_mismatch,
        "near_boundary_values_abs_0_to_0_25": near_boundary_values,
        "minimum_nonzero_abs": None if min_nonzero_abs is math.inf else min_nonzero_abs,
        "first_failure": first_failure,
        "raw_sha256": raw_hash.hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    watched = [ROOT / "submission.zip", ROOT / "all_scores.csv", ROOT / "best_score.json"]
    hashes_before = {str(path.relative_to(ROOT)): file_digest(path) for path in watched}
    data = CANDIDATE.read_bytes()
    model = onnx.load_model_from_string(data)

    static, inferred = static_audit(model)
    runtime_shapes = runtime_shape_audit(inferred)
    print("PASS strict static and runtime truthful-shape audit", flush=True)
    official_profile = profile(model)
    print(f"PASS official gold profile: {official_profile}", flush=True)
    margin_ok, min_margin = scoring.model_margin_stable(model, 71, margin=0.25)
    if not margin_ok:
        raise RuntimeError(f"visible margin gate failed: min={min_margin}")
    print(f"PASS visible margin: min_nonzero_abs={min_margin}", flush=True)

    fresh = [fresh_audit(model, seed) for seed in FRESH_SEEDS]
    fresh_pass = all(
        run["right"] == FRESH_CASES_PER_SEED
        and run["wrong"] == 0
        and run["runtime_errors"] == 0
        and run["nonfinite_cases"] == 0
        and run["shape_mismatch_cases"] == 0
        and run["near_boundary_values_abs_0_to_0_25"] == 0
        for run in fresh
    )
    if not fresh_pass:
        raise RuntimeError(f"fresh gate failed: {fresh}")

    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_member = archive.read("task071.onnx")
    hashes_after = {str(path.relative_to(ROOT)): file_digest(path) for path in watched}
    evidence = {
        "task": 71,
        "policy": "GOLD_EXACT_AND_MARGIN_AND_FRESH_2000_X2_100_PERCENT",
        "authority": {
            "zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
            "zip_sha256_at_finish": file_digest(AUTHORITY_ZIP),
            "member_sha256_at_finish": digest(authority_member),
            "reported_cost_from_all_scores": 185,
            "known_shape_cloak": True,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": digest(data),
            "official_gold_exact": bool(official_profile["correct"]),
            "official_profile": official_profile,
            "visible_margin_stable": margin_ok,
            "visible_min_nonzero_abs": min_margin,
        },
        "static": static,
        "runtime_shape_audit": runtime_shapes,
        "fresh": fresh,
        "fresh_total": {
            "cases": sum(run["cases"] for run in fresh),
            "right": sum(run["right"] for run in fresh),
            "wrong": sum(run["wrong"] for run in fresh),
        },
        "fresh_gate_pass": fresh_pass,
        "admitted_as_improvement": int(official_profile["cost"]) < 185,
        "rejection_reason": (
            None
            if int(official_profile["cost"]) < 185
            else (
                f"truthful fully-gated equivalent costs {official_profile['cost']}, "
                "above shape-cloaked authority cost 185"
            )
        ),
        "root_hashes_before": hashes_before,
        "root_hashes_after": hashes_after,
        "root_hashes_unchanged_during_audit": hashes_before == hashes_after,
        "root_or_checkpoint_modified_by_this_script": False,
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS fresh {evidence['fresh_total']['right']}/{evidence['fresh_total']['cases']}")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
