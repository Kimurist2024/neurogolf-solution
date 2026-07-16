#!/usr/bin/env python3
"""Independent policy95 audit for the task070 cost-50 historical candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATES = {
    "cost50": ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task070_r01_static50.onnx",
    "cost52": ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task070_r02_static52.onnx",
    "cost53": ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task070_r03_static53.onnx",
    "cost56": ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task070_r04_static56.onnx",
    "cost58": ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task070_r05_static58.onnx",
}
AUTHORITY_ZIP = ROOT / "submission_base_8011.05.zip"
AUTHORITY_ZIP_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
TASK = 70
SEEDS = (303_070_101, 303_070_202)
FRESH_PER_SEED = 2000
POLICY_RATE = 0.95

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim]


def structure(model: onnx.ModelProto) -> dict[str, object]:
    reasons = []
    try:
        onnx.checker.check_model(model, full_check=True)
        full_check = True
    except Exception as exc:
        full_check = False
        reasons.append(f"full_check:{type(exc).__name__}:{exc}")
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        strict_shape = True
    except Exception as exc:
        strict_shape = False
        reasons.append(f"strict_shape:{type(exc).__name__}:{exc}")
    if model.functions:
        reasons.append("local_functions")
    if model.graph.sparse_initializer:
        reasons.append("sparse_initializer")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        reasons.append("noncanonical_io_count")
    elif dims(model.graph.input[0]) != [1, 10, 30, 30] or dims(model.graph.output[0]) != [1, 10, 30, 30]:
        reasons.append("noncanonical_io_shape")
    graph_outputs = {value.name for value in model.graph.output}
    intermediate_outputs = []
    for node in model.graph.node:
        if node.domain or node.op_type in BANNED or "Sequence" in node.op_type:
            reasons.append(f"domain_or_banned:{node.domain}:{node.op_type}")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                reasons.append("nested_graph")
        intermediate_outputs.extend(name for name in node.output if name and name not in graph_outputs)
    nonfinite_initializers = []
    for init in model.graph.initializer:
        array = onnx.numpy_helper.to_array(init)
        if not np.all(np.isfinite(array)):
            nonfinite_initializers.append(init.name)
    if nonfinite_initializers:
        reasons.append("nonfinite_initializers")
    return {
        "full_check": full_check,
        "strict_shape_inference_data_prop": strict_shape,
        "input_shape": dims(model.graph.input[0]) if model.graph.input else None,
        "output_shape": dims(model.graph.output[0]) if model.graph.output else None,
        "node_count": len(model.graph.node),
        "ops": [node.op_type for node in model.graph.node],
        "einsum_fanin": [len(node.input) for node in model.graph.node if node.op_type == "Einsum"],
        "intermediate_output_count": len(intermediate_outputs),
        "intermediate_outputs": intermediate_outputs,
        "nonfinite_initializers": nonfinite_initializers,
        "conv_bias_ub": False,
        "reasons": reasons,
        "safe": full_check and strict_shape and not reasons,
    }


def session(model: onnx.ModelProto, optimization: str, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disabled" else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(sanitized.SerializeToString(), options,
                                providers=["CPUExecutionProvider"])


def cases_known() -> list[dict[str, np.ndarray]]:
    result = []
    examples = scoring.load_examples(TASK)
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is not None:
                result.append(benchmark)
    return result


def cases_fresh(seed: int) -> tuple[list[dict[str, np.ndarray]], int]:
    task_hash = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())[f"{TASK:03d}"]
    generator = importlib.import_module(f"task_{task_hash}")
    random.seed(seed)
    result = []
    attempts = 0
    while len(result) < FRESH_PER_SEED and attempts < FRESH_PER_SEED * 4:
        attempts += 1
        try:
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
        except Exception:
            continue
        if benchmark is not None:
            result.append(benchmark)
    return result, attempts


def evaluate(sess: ort.InferenceSession, cases: list[dict[str, np.ndarray]]) -> dict[str, object]:
    right = wrong = errors = nonfinite_cases = shape_mismatches = small_positive = 0
    minimum_positive = math.inf
    first_wrong = first_error = None
    for index, benchmark in enumerate(cases):
        try:
            raw = scoring._raw_output(sess, benchmark["input"])
        except Exception as exc:
            errors += 1
            if first_error is None:
                first_error = {"index": index, "type": type(exc).__name__, "message": str(exc)}
            continue
        if tuple(raw.shape) != (1, 10, 30, 30):
            shape_mismatches += 1
        if not np.all(np.isfinite(raw)):
            nonfinite_cases += 1
            continue
        positives = raw[raw > 0]
        if positives.size:
            minimum_positive = min(minimum_positive, float(positives.min()))
            small_positive += int(np.count_nonzero((positives > 0) & (positives < 0.25)))
        predicted = (raw > 0).astype(np.float32)
        if np.array_equal(predicted, benchmark["output"]):
            right += 1
        else:
            wrong += 1
            if first_wrong is None:
                mismatch = np.argwhere(predicted != benchmark["output"])
                first_wrong = {"index": index, "mismatch_cells": int(mismatch.shape[0])}
    total = right + wrong + errors + nonfinite_cases
    return {
        "total": total, "right": right, "wrong": wrong, "errors": errors,
        "nonfinite_cases": nonfinite_cases, "shape_mismatches": shape_mismatches,
        "small_positive_elements_0_to_0_25": small_positive,
        "minimum_positive": None if minimum_positive is math.inf else minimum_positive,
        "accuracy": right / total if total else None,
        "first_wrong": first_wrong, "first_error": first_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(CANDIDATES), default="cost52")
    args = parser.parse_args()
    candidate = CANDIDATES[args.variant]
    out = HERE / f"task070_policy95_{args.variant}_audit.json"
    authority_bytes = AUTHORITY_ZIP.read_bytes()
    if digest(authority_bytes) != AUTHORITY_ZIP_SHA256:
        raise RuntimeError("authority ZIP SHA mismatch")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_model = onnx.load_model_from_string(archive.read("task070.onnx"))
    candidate_model = onnx.load(candidate)
    with tempfile.TemporaryDirectory(prefix="task070_policy95_", dir="/tmp") as tmp:
        authority_profile = scoring.score_and_verify(authority_model, TASK, tmp, label="authority", require_correct=False)
        candidate_profile = scoring.score_and_verify(candidate_model, TASK, tmp, label="candidate", require_correct=False)
    stable, margin = scoring.model_margin_stable(candidate_model, TASK)
    known = cases_known()
    fresh_sets = {}
    fresh_attempts = {}
    for seed in SEEDS:
        fresh_sets[seed], fresh_attempts[seed] = cases_fresh(seed)

    configurations = [(mode, threads) for mode in ("disabled", "default") for threads in (1, 4)]
    known_rows = []
    fresh_rows = []
    for mode, threads in configurations:
        try:
            sess = session(candidate_model, mode, threads)
        except Exception as exc:
            fail = {"optimization": mode, "threads": threads,
                    "session_error": f"{type(exc).__name__}:{exc}"}
            known_rows.append(fail)
            for seed in SEEDS:
                fresh_rows.append({**fail, "seed": seed})
            continue
        known_rows.append({"optimization": mode, "threads": threads, **evaluate(sess, known)})
        for seed in SEEDS:
            fresh_rows.append({"optimization": mode, "threads": threads, "seed": seed,
                               "generation_attempts": fresh_attempts[seed],
                               **evaluate(sess, fresh_sets[seed])})

    structural = structure(candidate_model)
    known_pass = all(
        not row.get("session_error") and row["wrong"] == row["errors"] == row["nonfinite_cases"] == 0
        and row["shape_mismatches"] == row["small_positive_elements_0_to_0_25"] == 0
        for row in known_rows
    )
    fresh_pass = all(
        not row.get("session_error") and row["errors"] == row["nonfinite_cases"] == 0
        and row["shape_mismatches"] == row["small_positive_elements_0_to_0_25"] == 0
        and row["accuracy"] >= POLICY_RATE
        for row in fresh_rows
    )
    actual_lower = bool(candidate_profile and authority_profile and
                        candidate_profile["cost"] < authority_profile["cost"])
    policy95 = bool(structural["safe"] and known_pass and fresh_pass and actual_lower and stable)
    payload = {
        "task": TASK,
        "authority": {"zip": AUTHORITY_ZIP.name, "zip_sha256": AUTHORITY_ZIP_SHA256,
                      "model_sha256": digest(authority_model.SerializeToString()),
                      "profile": authority_profile},
        "candidate": {"path": str(candidate.relative_to(ROOT)),
                      "sha256": digest(candidate.read_bytes()), "profile": candidate_profile},
        "cost_delta": None if not (authority_profile and candidate_profile)
                      else int(authority_profile["cost"]) - int(candidate_profile["cost"]),
        "score_gain": None if not (authority_profile and candidate_profile)
                      else math.log(authority_profile["cost"] / candidate_profile["cost"]),
        "structure": structural,
        "margin_stable": bool(stable), "margin_min": margin,
        "known_case_count": len(known), "known_four_configs": known_rows,
        "fresh_policy_rate": POLICY_RATE, "fresh_seeds": list(SEEDS),
        "fresh_per_seed": FRESH_PER_SEED, "fresh_four_configs_two_seeds": fresh_rows,
        "known_pass": known_pass, "fresh_pass": fresh_pass,
        "actual_strict_lower": actual_lower,
        "classification": "POLICY95_PRIVATE_ZERO_RISK" if policy95 else "REJECT",
        "admit_policy95": policy95,
        "guaranteed_safe": False,
        "protected_writes": "none; evidence directory only",
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"classification": payload["classification"],
                      "cost_delta": payload["cost_delta"], "score_gain": payload["score_gain"],
                      "known_pass": known_pass, "fresh_pass": fresh_pass,
                      "evidence": str(out.relative_to(ROOT))}, indent=2))
    return 0 if policy95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
