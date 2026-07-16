#!/usr/bin/env python3
"""Fail-closed rebase audit for the exact task175 gauge reduction.

This lane is evidence-only: it never changes the root submission, score CSV,
or score pointer.  The audit pins the 8018.91 archive/member and requires full
ONNX checking, canonical static shapes, both local and official known gold,
stable raw margin, and two independent 2,000-case generator runs at 100%.
"""

from __future__ import annotations

import hashlib
import importlib
import json
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
AUTHORITY = ROOT / "submission_base_8018.91.zip"
CANDIDATE = HERE / "candidates" / "task175_gauge_remove_w_v.onnx"
EVIDENCE = HERE / "task175_evidence.json"

AUTHORITY_SHA256 = "e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091"
MEMBER_SHA256 = "b6404486ccc1a74c36bab6031f11c54c7326f787a743f02dff77e63c782af343"
CANDIDATE_SHA256 = "acead77ce6b60ae5d5dd88e5c2c006cecdac6c9c5fd56bc97b56b37b72df8a1a"
EXPECTED_COST = 134
SEEDS = (801_891_175, 801_891_176)
FRESH_K = 2_000
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from golf.rank_dir import cost_of  # noqa: E402
from golf import try_candidate as try_mod  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else -1
            for dim in value.type.tensor_type.shape.dim]


def strict_structure(model: onnx.ModelProto) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    reasons: list[str] = []
    if model.functions:
        reasons.append("local_functions")
    if model.graph.sparse_initializer:
        reasons.append("sparse_initializer")
    if len(inferred.graph.input) != 1 or len(inferred.graph.output) != 1:
        reasons.append("noncanonical_io_count")
    elif dims(inferred.graph.input[0]) != [1, 10, 30, 30] or dims(inferred.graph.output[0]) != [1, 10, 30, 30]:
        reasons.append("noncanonical_io_shape")
    for value in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        if value.type.HasField("tensor_type") and any(dim <= 0 for dim in dims(value)):
            reasons.append(f"nonstatic:{value.name}")
    for node in inferred.graph.node:
        if node.op_type in BANNED or "Sequence" in node.op_type:
            reasons.append(f"banned:{node.op_type}")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                reasons.append("nested_graph")
    for initializer in inferred.graph.initializer:
        values = onnx.numpy_helper.to_array(initializer)
        if not np.all(np.isfinite(values)):
            reasons.append(f"nonfinite_initializer:{initializer.name}")
    return {
        "pass": not reasons,
        "reasons": sorted(set(reasons)),
        "input_shape": dims(inferred.graph.input[0]),
        "output_shape": dims(inferred.graph.output[0]),
        "ops": [node.op_type for node in inferred.graph.node],
    }


def raw_session(model: onnx.ModelProto, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected candidate")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known_audit(model: onnx.ModelProto) -> dict[str, object]:
    session = raw_session(model, 1)
    total = failures = small_positive = 0
    min_positive = float("inf")
    max_nonpositive = -float("inf")
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(175)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            raw = scoring._raw_output(session, benchmark["input"])
            predicted = raw > 0.0
            expected = benchmark["output"] > 0.0
            failures += int(not np.array_equal(predicted, expected))
            small_positive += int(np.count_nonzero((raw > 0.0) & (raw < 0.25)))
            if np.any(predicted):
                min_positive = min(min_positive, float(raw[predicted].min()))
            if np.any(~predicted):
                max_nonpositive = max(max_nonpositive, float(raw[~predicted].max()))
            total += 1
    official, mismatch = try_mod._verify_gold(model, 175)
    return {
        "total": total,
        "failures": failures,
        "local_gold_exact": failures == 0,
        "official_gold_exact": bool(official),
        "official_mismatch": mismatch,
        "small_positive_count": small_positive,
        "min_positive_raw": min_positive,
        "max_nonpositive_raw": max_nonpositive,
    }


def fresh_audit(candidate: onnx.ModelProto, authority: onnx.ModelProto, seed: int) -> dict[str, object]:
    generator = importlib.import_module("task_73251a56")
    candidate_sessions = [raw_session(candidate, threads) for threads in (1, 4)]
    authority_session = raw_session(authority, 1)
    random.seed(seed)
    total = failures = authority_sign_differences = runtime_errors = nonfinite = shape_errors = 0
    small_positive = 0
    min_positive = float("inf")
    max_nonpositive = -float("inf")
    while total < FRESH_K:
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        expected = benchmark["output"] > 0.0
        input_array = benchmark["input"]
        try:
            raws = [session.run(["output"], {"input": input_array})[0]
                    for session in candidate_sessions]
            authority_raw = authority_session.run(["output"], {"input": input_array})[0]
        except Exception:
            runtime_errors += 1
            total += 1
            continue
        for raw in raws:
            if raw.shape != (1, 10, 30, 30):
                shape_errors += 1
            if not np.all(np.isfinite(raw)):
                nonfinite += 1
            predicted = raw > 0.0
            failures += int(not np.array_equal(predicted, expected))
            small_positive += int(np.count_nonzero((raw > 0.0) & (raw < 0.25)))
            if np.any(predicted):
                min_positive = min(min_positive, float(raw[predicted].min()))
            if np.any(~predicted):
                max_nonpositive = max(max_nonpositive, float(raw[~predicted].max()))
        authority_sign_differences += int(
            not np.array_equal(raws[0] > 0.0, authority_raw > 0.0)
        )
        total += 1
    return {
        "seed": seed,
        "cases": total,
        "ort_configurations_per_case": len(candidate_sessions),
        "candidate_failures_across_configurations": failures,
        "authority_sign_difference_cases": authority_sign_differences,
        "runtime_errors": runtime_errors,
        "nonfinite_outputs": nonfinite,
        "shape_errors": shape_errors,
        "small_positive_count": small_positive,
        "min_positive_raw": min_positive,
        "max_nonpositive_raw": max_nonpositive,
        "pass": failures == runtime_errors == nonfinite == shape_errors == small_positive == authority_sign_differences == 0,
    }


def main() -> int:
    authority_blob = AUTHORITY.read_bytes()
    candidate_blob = CANDIDATE.read_bytes()
    if sha256(authority_blob) != AUTHORITY_SHA256:
        raise RuntimeError("8018.91 authority drift")
    if sha256(candidate_blob) != CANDIDATE_SHA256:
        raise RuntimeError("candidate drift")
    with zipfile.ZipFile(AUTHORITY) as archive:
        member_blob = archive.read("task175.onnx")
    if sha256(member_blob) != MEMBER_SHA256:
        raise RuntimeError("task175 authority member drift")

    candidate = onnx.load_model_from_string(candidate_blob)
    authority = onnx.load_model_from_string(member_blob)
    structure = strict_structure(candidate)
    known = known_audit(candidate)
    with tempfile.TemporaryDirectory(prefix="restart8018_task175_", dir="/tmp") as tempdir:
        profile = scoring.score_and_verify(candidate, 175, tempdir, label="task175", require_correct=False)
    memory, params, cost = cost_of(str(CANDIDATE))
    fresh = [fresh_audit(candidate, authority, seed) for seed in SEEDS]
    accepted = bool(
        structure["pass"]
        and known["local_gold_exact"]
        and known["official_gold_exact"]
        and known["small_positive_count"] == 0
        and profile["correct"]
        and int(profile["cost"]) == int(cost) == EXPECTED_COST
        and all(item["pass"] for item in fresh)
    )
    result = {
        "policy": "official/local gold exact + full/static checker + stable margin + fresh 2000x2 100%",
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "sha256": AUTHORITY_SHA256,
            "member_sha256": MEMBER_SHA256,
            "cost": 140,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": CANDIDATE_SHA256,
            "memory": memory,
            "params": params,
            "cost": cost,
            "gain": float(np.log(140.0 / float(EXPECTED_COST))),
        },
        "structure": structure,
        "known": known,
        "official_profile": profile,
        "fresh": fresh,
        "accepted": accepted,
        "protected_root_files_modified": False,
    }
    EVIDENCE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
