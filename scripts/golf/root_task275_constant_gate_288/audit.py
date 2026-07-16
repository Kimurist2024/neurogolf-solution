#!/usr/bin/env python3
"""Primary all-support/pass-through audit for task275's constant gate."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = HERE / "candidates/task275_constant_gate.onnx"
CANDIDATE_SHA256 = "aa7d864004f360cd1ca4627afb747a1345a041d7aa371e81834109401d758a51"
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
AUTHORITY_MEMBER_SHA256 = "d31e860f7243a66917a06b306b6bd856da71a0f09dfc4d4d0e6472f1c9e2003f"
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
KNOWN_PATH = ROOT / "inputs/neurogolf-2026/task275.json"
SEEDS = (288_275_001, 288_375_001)
# The sole remaining Einsum has 41 inputs and is expensive.  The correctness
# claim is the complete one-hot invariant below; these disjoint streams are a
# runtime/configuration check, not the proof itself.
FRESH_COUNT = 500
EXPECTED = (1, 10, 30, 30)
CONFIGS = (
    ("disabled_t1", True, 1),
    ("default_t1", False, 1),
    ("disabled_t4", True, 4),
    ("default_t4", False, 4),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GENERATOR = importlib.import_module("task_b190f7f5")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cases_known() -> list[dict]:
    payload = json.loads(KNOWN_PATH.read_text())
    return [case for split in ("train", "test", "arc-gen") for case in payload[split]]


def cases_fresh(seed: int) -> list[dict]:
    random.seed(seed)
    return [GENERATOR.generate() for _ in range(FRESH_COUNT)]


def make_session(data: bytes, disabled: bool, threads: int) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def evaluate(
    authority: ort.InferenceSession,
    candidate: ort.InferenceSession,
    cases: list[dict],
) -> dict:
    right = raw_equal = sign_equal = errors = nonfinite = shape = 0
    onehot_failures = sum_failures = small_positive = 0
    minimum_positive = math.inf
    candidate_raw_hash = hashlib.sha256()
    first_failure = None
    for index, case in enumerate(cases):
        converted = scoring.convert_to_numpy(case)
        if converted is None:
            errors += 1
            continue
        value = np.asarray(converted["input"])
        onehot = np.sum(value, axis=1)
        if not np.array_equal(onehot, np.ones((1, 30, 30), dtype=onehot.dtype)):
            onehot_failures += 1
        if float(np.sum(value, dtype=np.float32)) != 900.0:
            sum_failures += 1
        try:
            source_raw = authority.run(["output"], {"input": value})[0]
            candidate_raw = candidate.run(["output"], {"input": value})[0]
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = {"index": index, "kind": "runtime", "error": repr(exc)}
            continue
        shape += int(tuple(source_raw.shape) != EXPECTED or tuple(candidate_raw.shape) != EXPECTED)
        nonfinite += int(not np.isfinite(source_raw).all() or not np.isfinite(candidate_raw).all())
        source_sign = source_raw > 0.0
        candidate_sign = candidate_raw > 0.0
        exact_raw = np.array_equal(source_raw.view(np.uint8), candidate_raw.view(np.uint8))
        exact_sign = np.array_equal(source_sign, candidate_sign)
        raw_equal += int(exact_raw)
        sign_equal += int(exact_sign)
        right += int(np.array_equal(candidate_sign, converted["output"] > 0.0))
        positive = candidate_raw[candidate_raw > 0.0]
        if positive.size:
            minimum_positive = min(minimum_positive, float(np.min(positive)))
            small_positive += int(np.count_nonzero(positive < 0.25))
        candidate_raw_hash.update(np.ascontiguousarray(candidate_raw).tobytes())
        if first_failure is None and (not exact_raw or not exact_sign):
            first_failure = {"index": index, "kind": "authority_mismatch"}
    total = len(cases)
    return {
        "total": total,
        "right": right,
        "accuracy": right / total,
        "raw_equal": raw_equal,
        "sign_equal": sign_equal,
        "errors": errors,
        "nonfinite_cases": nonfinite,
        "shape_mismatches": shape,
        "onehot_failures": onehot_failures,
        "float32_sum_900_failures": sum_failures,
        "small_positive_elements": small_positive,
        "minimum_positive": None if math.isinf(minimum_positive) else minimum_positive,
        "candidate_raw_sha256": candidate_raw_hash.hexdigest(),
        "first_failure": first_failure,
    }


def profile(data: bytes, label: str) -> dict:
    path = HERE / f"{label}.onnx.audit-copy"
    path.write_bytes(data)
    memory, params, cost = cost_of(str(path))
    path.unlink()
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    started = time.time()
    candidate_data = CANDIDATE.read_bytes()
    authority_zip = AUTHORITY.read_bytes()
    if digest(candidate_data) != CANDIDATE_SHA256:
        raise RuntimeError("candidate hash mismatch")
    if digest(authority_zip) != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP hash mismatch")
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read("task275.onnx")
    if digest(authority_data) != AUTHORITY_MEMBER_SHA256:
        raise RuntimeError("authority member hash mismatch")

    source = onnx.load_model_from_string(authority_data)
    candidate = onnx.load_model_from_string(candidate_data)
    onnx.checker.check_model(copy.deepcopy(candidate), full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(candidate), strict_mode=True, data_prop=True
    )
    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in candidate.graph.initializer}
    source_arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in source.graph.initializer}
    unchanged_initializers = all(
        name in arrays and source_arrays[name].dtype == arrays[name].dtype
        and source_arrays[name].shape == arrays[name].shape
        and source_arrays[name].tobytes() == arrays[name].tobytes()
        for name in source_arrays if name not in {"GW", "GB"}
    )
    gate_exact = (
        arrays["gate_const"].dtype == np.float32
        and arrays["gate_const"].shape == (1, 2, 1, 1)
        and np.array_equal(
            arrays["gate_const"],
            np.asarray([-875.0, 7.0], dtype=np.float32).reshape(1, 2, 1, 1),
        )
    )
    node = candidate.graph.node[0]
    structure = {
        "full_check": True,
        "strict_data_prop": True,
        "input_shape": [int(dim.dim_value) for dim in inferred.graph.input[0].type.tensor_type.shape.dim],
        "output_shape": [int(dim.dim_value) for dim in inferred.graph.output[0].type.tensor_type.shape.dim],
        "single_output_einsum": len(candidate.graph.node) == 1 and node.op_type == "Einsum" and list(node.output) == ["output"],
        "einsum_inputs": len(node.input),
        "inherited_giant_einsum": len(node.input) == 41,
        "all_non_gate_initializers_byte_identical": unchanged_initializers,
        "removed_initializers": sorted(set(source_arrays) - set(arrays)),
        "added_initializers": sorted(set(arrays) - set(source_arrays)),
        "gate_exact": gate_exact,
        "conv_bias_findings": check_conv_bias(candidate),
        "functions": len(candidate.functions),
        "sparse_initializers": len(candidate.graph.sparse_initializer),
        "standard_domains": all(item.domain in ("", "ai.onnx") for item in candidate.opset_import)
        and all(item.domain in ("", "ai.onnx") for item in candidate.graph.node),
    }
    structure["pass"] = bool(
        structure["input_shape"] == list(EXPECTED)
        and structure["output_shape"] == list(EXPECTED)
        and structure["single_output_einsum"]
        and structure["inherited_giant_einsum"]
        and structure["all_non_gate_initializers_byte_identical"]
        and structure["removed_initializers"] == ["GB", "GW"]
        and structure["added_initializers"] == ["gate_const"]
        and structure["gate_exact"]
        and not structure["conv_bias_findings"]
        and structure["functions"] == 0
        and structure["sparse_initializers"] == 0
        and structure["standard_domains"]
    )

    corpora = [("known", cases_known())]
    corpora.extend((f"fresh_{seed}", cases_fresh(seed)) for seed in SEEDS)
    runs = []
    for label, disabled, threads in CONFIGS:
        source_run = make_session(authority_data, disabled, threads)
        candidate_run = make_session(candidate_data, disabled, threads)
        runs.append({
            "config": label,
            "corpora": {name: evaluate(source_run, candidate_run, cases) for name, cases in corpora},
        })
    reference = runs[0]["corpora"]
    config_raw_stable = all(
        row["candidate_raw_sha256"] == reference[name]["candidate_raw_sha256"]
        for run in runs for name, row in run["corpora"].items()
    )
    all_rows = [row for run in runs for row in run["corpora"].values()]
    candidate_profile = profile(candidate_data, "candidate")
    authority_profile = profile(authority_data, "authority")
    passed = bool(
        structure["pass"]
        and authority_profile == {"memory": 12, "params": 416, "cost": 428}
        and candidate_profile == {"memory": 0, "params": 414, "cost": 414}
        and config_raw_stable
        and all(row["raw_equal"] == row["total"] for row in all_rows)
        and all(row["sign_equal"] == row["total"] for row in all_rows)
        and all(row["errors"] == 0 for row in all_rows)
        and all(row["nonfinite_cases"] == 0 for row in all_rows)
        and all(row["shape_mismatches"] == 0 for row in all_rows)
        and all(row["onehot_failures"] == 0 for row in all_rows)
        and all(row["float32_sum_900_failures"] == 0 for row in all_rows)
    )
    evidence = {
        "status": "PASS_EXACT_ALL_SUPPORT_PRIMARY" if passed else "REJECT",
        "classification": "EXACT_CANONICAL_ONEHOT_GATE_CONSTANT_FOLD_INHERITED_GIANT_PASS_THROUGH",
        "proof": {
            "canonical_support": "scoring input is one-hot [1,10,30,30], so exactly one channel is1 at every one of900 cells",
            "sum": "ReduceSum over channel/height/width is exactly float32 900",
            "conv": "1x1 Conv with GW=[-1,0] and GB=[25,7] is exactly [-875,7]",
            "downstream": "the sole downstream 41-input Einsum and every other initializer are unchanged",
            "giant_risk": "the 41-input Einsum is inherited byte-for-byte except gate input name; the candidate removes two prefix nodes and introduces no additional runtime structure",
        },
        "authority": {"zip_sha256": digest(authority_zip), "member_sha256": digest(authority_data), "profile": authority_profile},
        "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": digest(candidate_data), "profile": candidate_profile, "gain": math.log(428 / 414)},
        "structure": structure,
        "runs": runs,
        "config_raw_stable": config_raw_stable,
        "audit_pass": passed,
        "elapsed_seconds": time.time() - started,
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({
        "status": evidence["status"],
        "profiles": {"authority": authority_profile, "candidate": candidate_profile},
        "runs": [{"config": run["config"], "corpora": {
            name: {key: row[key] for key in ("right", "total", "accuracy", "raw_equal", "errors", "nonfinite_cases", "shape_mismatches", "onehot_failures", "float32_sum_900_failures", "minimum_positive")}
            for name, row in run["corpora"].items()
        }} for run in runs],
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
