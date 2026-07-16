#!/usr/bin/env python3
"""Primary four-configuration POLICY90 audit for task355 cost 249."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, numpy_helper

import screen


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task355_r04_static249.onnx"
EXPECTED_SHA256 = "7ca617858a19310a433010e6e50da46b4d562d76f3d0688665c8387bdf6f24d8"
FRESH_SEEDS = (283_555_001, 283_655_001)
FRESH_COUNT = 10_000
CONFIGS = (
    ("disabled_t1", True, 1),
    ("default_t1", False, 1),
    ("disabled_t4", True, 4),
    ("default_t4", False, 4),
)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim]


def make_session(model: onnx.ModelProto, disabled: bool, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected candidate")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def evaluate(run: ort.InferenceSession, cases: list[dict]) -> dict:
    right = wrong = errors = nonfinite = shape = small_positive = 0
    minimum_positive = math.inf
    signs = hashlib.sha256()
    raw_hash = hashlib.sha256()
    first_failure = None
    for index, case in enumerate(cases):
        converted = scoring.convert_to_numpy(case)
        if converted is None:
            errors += 1
            continue
        try:
            raw = run.run(["output"], {"input": converted["input"]})[0]
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = {"index": index, "kind": "runtime", "error": repr(exc)}
            continue
        if tuple(raw.shape) != screen.EXPECTED:
            shape += 1
        if not np.isfinite(raw).all():
            nonfinite += 1
        positive = raw[raw > 0.0]
        if positive.size:
            minimum_positive = min(minimum_positive, float(np.min(positive)))
            small_positive += int(np.count_nonzero(positive < 0.25))
        actual = np.asarray(raw > 0.0, dtype=np.uint8)
        expected = np.asarray(converted["output"] > 0.0, dtype=np.uint8)
        signs.update(np.packbits(actual).tobytes())
        raw_hash.update(np.ascontiguousarray(raw).tobytes())
        if np.array_equal(actual, expected):
            right += 1
        else:
            wrong += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "kind": "wrong",
                    "differing_cells": int(np.count_nonzero(actual != expected)),
                }
    total = len(cases)
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "total": total,
        "accuracy": right / total,
        "policy90": right / total >= 0.90,
        "nonfinite_cases": nonfinite,
        "shape_mismatches": shape,
        "small_positive_elements": small_positive,
        "minimum_positive": None if math.isinf(minimum_positive) else minimum_positive,
        "sign_sha256": signs.hexdigest(),
        "raw_sha256": raw_hash.hexdigest(),
        "first_failure": first_failure,
    }


def nested_graphs(model: onnx.ModelProto) -> int:
    total = 0
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                total += 1
            elif attribute.type == AttributeProto.GRAPHS:
                total += len(attribute.graphs)
    return total


def static_audit(model: onnx.ModelProto) -> dict:
    onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }
    node_outputs = [name for node in inferred.graph.node for name in node.output if name]
    missing = [name for name in node_outputs if name not in typed]
    # Rank-0 tensors are valid static scalars; only an unknown/non-positive
    # declared dimension is a structural failure.
    nonstatic = [name for name, value in typed.items()
                 if any(dim is None or dim <= 0 for dim in dims(value))]
    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    ops = Counter(node.op_type for node in model.graph.node)
    domains = sorted({domain for domain in [
        *(item.domain for item in model.opset_import),
        *(node.domain for node in model.graph.node),
    ] if domain not in ("", "ai.onnx")})
    explicit_lookup = sorted({node.op_type for node in model.graph.node
                              if node.op_type in {"TfIdfVectorizer", "Hardmax", "Gather", "GatherND", "ScatterND"}})
    giant_initializers = [name for name, array in arrays.items() if array.size >= 10_000]
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    external = [item.name for item in model.graph.initializer
                if item.data_location == TensorProto.EXTERNAL or item.external_data]
    return {
        "full_check": True,
        "strict_data_prop": True,
        "input_shape": dims(inferred.graph.input[0]),
        "output_shape": dims(inferred.graph.output[0]),
        "canonical_static_io": dims(inferred.graph.input[0]) == list(screen.EXPECTED)
        and dims(inferred.graph.output[0]) == list(screen.EXPECTED),
        "all_node_outputs_static_positive": not missing and not nonstatic,
        "missing_typed_node_outputs": missing,
        "nonstatic_typed_values": nonstatic,
        "node_count": len(model.graph.node),
        "op_histogram": dict(sorted(ops.items())),
        "finite_initializers": all(np.isfinite(array).all() for array in arrays.values()),
        "initializer_elements": sum(int(array.size) for array in arrays.values()),
        "standard_domains": not domains,
        "nonstandard_domains": domains,
        "banned_ops": sorted(op for op in ops if op in BANNED or "Sequence" in op),
        "nested_graphs": nested_graphs(model),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": external,
        "explicit_lookup_ops": explicit_lookup,
        "giant_initializers": giant_initializers,
        "max_einsum_inputs": max_einsum,
        "conv_bias_findings": check_conv_bias(model),
    }


def main() -> None:
    started = time.time()
    candidate_data = CANDIDATE.read_bytes()
    if screen.digest(candidate_data) != EXPECTED_SHA256:
        raise RuntimeError("candidate hash mismatch")
    authority_data = screen.AUTHORITY_ZIP.read_bytes()
    if screen.digest(authority_data) != screen.AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP hash mismatch")
    with zipfile.ZipFile(screen.AUTHORITY_ZIP) as archive:
        authority_member = archive.read("task355.onnx")

    candidate = onnx.load_model_from_string(candidate_data)
    authority = onnx.load_model_from_string(authority_member)
    structure = static_audit(candidate)
    authority_tmp = HERE / "authority_task355.onnx.audit-copy"
    authority_tmp.write_bytes(authority_member)
    authority_cost = cost_of(str(authority_tmp))
    authority_tmp.unlink()
    candidate_cost = cost_of(str(CANDIDATE))

    authority_nodes = {tuple(node.output): node for node in authority.graph.node}
    candidate_nodes = {tuple(node.output): node for node in candidate.graph.node}
    removed = sorted(set(authority_nodes) - set(candidate_nodes))
    changed = []
    for output in sorted(set(authority_nodes) & set(candidate_nodes)):
        before = authority_nodes[output]
        after = candidate_nodes[output]
        if before.SerializeToString() != after.SerializeToString():
            changed.append({
                "output": list(output),
                "before": {"op": before.op_type, "inputs": list(before.input)},
                "after": {"op": after.op_type, "inputs": list(after.input)},
            })

    random.seed(FRESH_SEEDS[0])
    fresh_a = [screen.GENERATOR.generate() for _ in range(FRESH_COUNT)]
    random.seed(FRESH_SEEDS[1])
    fresh_b = [screen.GENERATOR.generate() for _ in range(FRESH_COUNT)]
    corpora = [("known", screen.cases_known()),
               (f"fresh_{FRESH_SEEDS[0]}", fresh_a),
               (f"fresh_{FRESH_SEEDS[1]}", fresh_b)]
    runs = []
    for label, disabled, threads in CONFIGS:
        run = make_session(candidate, disabled, threads)
        runs.append({
            "config": label,
            "corpora": {name: evaluate(run, cases) for name, cases in corpora},
        })

    reference = runs[0]["corpora"]
    config_sign_stable = all(
        row["sign_sha256"] == reference[name]["sign_sha256"]
        for run in runs for name, row in run["corpora"].items()
    )
    all_rows = [row for run in runs for row in run["corpora"].values()]
    structural_pass = bool(
        structure["canonical_static_io"]
        and structure["all_node_outputs_static_positive"]
        and structure["finite_initializers"]
        and structure["standard_domains"]
        and not structure["banned_ops"]
        and structure["nested_graphs"] == 0
        and structure["functions"] == 0
        and structure["sparse_initializers"] == 0
        and not structure["external_initializers"]
        and not structure["explicit_lookup_ops"]
        and not structure["giant_initializers"]
        and structure["max_einsum_inputs"] < 15
        and not structure["conv_bias_findings"]
    )
    audit_pass = bool(
        structural_pass
        and tuple(authority_cost) == (228, 22, 250)
        and tuple(candidate_cost) == (227, 22, 249)
        and removed == [("WA",)]
        and changed == [{
            "output": ["tieval"],
            "before": {"op": "Mul", "inputs": ["cnt4", "WA"]},
            "after": {"op": "Mul", "inputs": ["cnt4", "h_num"]},
        }]
        and config_sign_stable
        and all(row["policy90"] for row in all_rows)
        and all(row["errors"] == 0 for row in all_rows)
        and all(row["nonfinite_cases"] == 0 for row in all_rows)
        and all(row["shape_mismatches"] == 0 for row in all_rows)
        and all(row["small_positive_elements"] == 0 for row in all_rows)
    )
    evidence = {
        "status": "PASS_POLICY90_PRIMARY" if audit_pass else "REJECT",
        "classification": "NORMAL_POLICY90_NOT_EXACT_OFFICIAL_CORRECTNESS",
        "public_overfit_risk_note": (
            "task355 appears in a public overfit-risk list, but not in the project's "
            "private-zero catalog. Admission therefore requires the normal POLICY90 gates "
            "and does not claim complete correctness."
        ),
        "authority": {
            "zip": str(screen.AUTHORITY_ZIP.relative_to(ROOT)),
            "zip_sha256": screen.digest(authority_data),
            "member_sha256": screen.digest(authority_member),
            "cost": {"memory": authority_cost[0], "params": authority_cost[1], "cost": authority_cost[2]},
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": screen.digest(candidate_data),
            "cost": {"memory": candidate_cost[0], "params": candidate_cost[1], "cost": candidate_cost[2]},
            "gain": math.log(candidate_cost[2] + 1) - math.log(candidate_cost[2]),
        },
        "graph_delta": {"removed_outputs": [list(item) for item in removed], "changed_nodes": changed},
        "structure": {**structure, "pass": structural_pass},
        "runs": runs,
        "config_sign_stable": config_sign_stable,
        "audit_pass": audit_pass,
        "elapsed_seconds": time.time() - started,
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({
        "status": evidence["status"],
        "candidate_cost": evidence["candidate"]["cost"],
        "gain": evidence["candidate"]["gain"],
        "runs": [{"config": run["config"], "corpora": {
            name: {key: row[key] for key in ("right", "total", "accuracy", "errors", "nonfinite_cases", "shape_mismatches", "small_positive_elements", "minimum_positive")}
            for name, row in run["corpora"].items()
        }} for run in runs],
    }, indent=2))
    if not audit_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
