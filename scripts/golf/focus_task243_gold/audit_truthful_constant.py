#!/usr/bin/env python3
"""Fail-closed gold, margin, and 2,000-fresh audit for task243."""

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
CANDIDATE = HERE / "candidates/task243_truthful_safe.onnx"
OUTPUT = HERE / "evidence.json"
AUTHORITY_ZIP = ROOT / "submission.zip"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
FRESH_SEED = 243_20260715
FRESH_CASES = 2000

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str | None:
    return digest(path.read_bytes()) if path.is_file() else None


def static_audit(model: onnx.ModelProto) -> dict[str, Any]:
    onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    tensors = [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]
    shapes: dict[str, list[int]] = {}
    for value in tensors:
        tensor_type = value.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value"):
                raise RuntimeError(f"non-static dimension: {value.name}")
            if dim.dim_value <= 0:
                raise RuntimeError(f"non-positive dimension: {value.name}")
            shape.append(int(dim.dim_value))
        shapes[value.name] = shape

    banned = {
        "LOOP",
        "SCAN",
        "NONZERO",
        "UNIQUE",
        "SCRIPT",
        "FUNCTION",
        "COMPRESS",
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
    return {
        "full_check": True,
        "strict_shape_inference_data_prop": True,
        "all_shapes_static_positive": True,
        "shapes": shapes,
        "banned_ops": bad_ops,
        "nested_graphs": nested,
        "functions": len(model.functions),
        "finite_initializers": finite,
        "shape_cloak": False,
    }


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
        sanitized.SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )


def fresh_audit(model: onnx.ModelProto) -> dict[str, Any]:
    generator = importlib.import_module("task_9edfc990")
    random.seed(FRESH_SEED)
    np.random.seed(FRESH_SEED & 0xFFFFFFFF)
    session = make_session(model)
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name

    right = wrong = runtime_errors = nonfinite = shape_mismatch = 0
    near_boundary_values = 0
    min_nonzero_abs = math.inf
    first_failure = None
    raw_hash = hashlib.sha256()
    size_counts: dict[str, int] = {}
    started = time.monotonic()

    for index in range(FRESH_CASES):
        example = generator.generate()
        size = len(example["input"])
        size_counts[str(size)] = size_counts.get(str(size), 0) + 1
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"fresh[{index}] unexpectedly exceeds 30x30")
        try:
            raw = session.run(
                [output_name], {input_name: benchmark["input"]}
            )[0]
        except BaseException as exc:  # fail closed for every runtime error
            runtime_errors += 1
            wrong += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
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
                    "grid_size": size,
                }

        if (index + 1) % 100 == 0:
            print(
                f"fresh progress {index + 1}/{FRESH_CASES}: "
                f"right={right} wrong={wrong}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    return {
        "seed": FRESH_SEED,
        "cases": FRESH_CASES,
        "right": right,
        "wrong": wrong,
        "accuracy": right / FRESH_CASES,
        "runtime_errors": runtime_errors,
        "nonfinite_cases": nonfinite,
        "shape_mismatch_cases": shape_mismatch,
        "near_boundary_values_abs_0_to_0_25": near_boundary_values,
        "minimum_nonzero_abs": (
            None if min_nonzero_abs is math.inf else min_nonzero_abs
        ),
        "first_failure": first_failure,
        "raw_sha256": raw_hash.hexdigest(),
        "grid_size_counts": size_counts,
        "elapsed_seconds": elapsed,
    }


def profile(model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="task243_truthful_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(model),
            243,
            workdir,
            label="truthful",
            require_correct=True,
        )
    if result is None:
        raise RuntimeError("official gold/scoring gate rejected candidate")
    return result


def main() -> int:
    ort.set_default_logger_severity(4)
    watched = [ROOT / "submission.zip", ROOT / "all_scores.csv", ROOT / "best_score.json"]
    hashes_before = {str(path.relative_to(ROOT)): file_digest(path) for path in watched}

    data = CANDIDATE.read_bytes()
    model = onnx.load_model_from_string(data)
    static = static_audit(model)
    print("PASS static truthful-shape audit", flush=True)
    official_profile = profile(model)
    print(f"PASS official gold profile: {official_profile}", flush=True)
    margin_ok, min_margin = scoring.model_margin_stable(model, 243, margin=0.25)
    if not margin_ok:
        raise RuntimeError(f"visible margin gate failed: min={min_margin}")
    print(f"PASS visible margin: min_nonzero_abs={min_margin}", flush=True)
    fresh = fresh_audit(model)

    fresh_pass = (
        fresh["right"] == FRESH_CASES
        and fresh["wrong"] == 0
        and fresh["runtime_errors"] == 0
        and fresh["nonfinite_cases"] == 0
        and fresh["shape_mismatch_cases"] == 0
        and fresh["near_boundary_values_abs_0_to_0_25"] == 0
    )
    if not fresh_pass:
        raise RuntimeError(f"fresh gate failed: {fresh}")

    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_member = archive.read("task243.onnx")
    hashes_after = {str(path.relative_to(ROOT)): file_digest(path) for path in watched}
    evidence = {
        "task": 243,
        "policy": "GOLD_EXACT_AND_MARGIN_AND_FRESH_2000_100_PERCENT",
        "authority": {
            "zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
            "zip_sha256_at_finish": file_digest(AUTHORITY_ZIP),
            "member_sha256_at_finish": digest(authority_member),
            "reported_cost_from_all_scores": 145,
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
        "fresh": fresh,
        "fresh_gate_pass": fresh_pass,
        "admitted_as_improvement": int(official_profile["cost"]) < 145,
        "rejection_reason": (
            None
            if int(official_profile["cost"]) < 145
            else (
                f"truthful fully-gated equivalent costs {official_profile['cost']}, "
                "above authority cost 145"
            )
        ),
        "root_hashes_before": hashes_before,
        "root_hashes_after": hashes_after,
        "root_hashes_unchanged_during_audit": hashes_before == hashes_after,
        "root_or_checkpoint_modified_by_this_script": False,
    }
    OUTPUT.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"PASS fresh {FRESH_CASES}/{FRESH_CASES}", flush=True)
    print(OUTPUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
