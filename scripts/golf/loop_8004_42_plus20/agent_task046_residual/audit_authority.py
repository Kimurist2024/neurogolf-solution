#!/usr/bin/env python3
"""Fail-closed residual audit of the current task046 authority.

This script does not emit or modify a candidate.  It characterizes the exact
authority member, runs the known and fresh sets in four ORT configurations,
and records raw-output and intermediate nonfinite signatures.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
MEMBER = "task046.onnx"
TASK = 46
EXPECTED_MEMBER_SHA256 = "fb649383229d5cdcb562b8c1ce52256ff344193810888b795c20ac0aa0660d77"
TASK_HASH = "234bbc79"
FRESH_SEED = 202607150046
FRESH_COUNT = 512
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.dim_param:
            result.append(dim.dim_param)
        else:
            result.append("?")
    return result


def type_size(elem_type: int) -> int:
    sizes = {
        TensorProto.BOOL: 1,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.FLOAT16: 2,
        TensorProto.BFLOAT16: 2,
        TensorProto.UINT32: 4,
        TensorProto.INT32: 4,
        TensorProto.FLOAT: 4,
        TensorProto.UINT64: 8,
        TensorProto.INT64: 8,
        TensorProto.DOUBLE: 8,
        TensorProto.COMPLEX64: 8,
        TensorProto.COMPLEX128: 16,
    }
    return sizes[elem_type]


def value_record(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor = value.type.tensor_type
    shape = tensor_shape(value)
    elements = math.prod(dim for dim in shape if isinstance(dim, int))
    static = all(isinstance(dim, int) and dim > 0 for dim in shape)
    return {
        "name": value.name,
        "dtype": TensorProto.DataType.Name(tensor.elem_type),
        "shape": shape,
        "static_positive": static,
        "bytes": elements * type_size(tensor.elem_type) if static else None,
    }


def nested_graph_count(model: onnx.ModelProto) -> int:
    count = 0
    stack = list(model.graph.node)
    while stack:
        node = stack.pop()
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                count += 1
                stack.extend(attr.g.node)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                count += len(attr.graphs)
                for graph in attr.graphs:
                    stack.extend(graph.node)
    return count


def structural_audit(data: bytes) -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, Any]]:
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
    }
    graph_outputs = {value.name for value in model.graph.output}
    node_outputs = [name for node in model.graph.node for name in node.output if name]
    inferred_records = [value_record(typed[name]) for name in node_outputs]
    memory = sum(
        int(row["bytes"])
        for row in inferred_records
        if row["name"] not in graph_outputs and row["bytes"] is not None
    )
    params = sum(int(numpy_helper.to_array(item).size) for item in model.graph.initializer)

    consumers: Counter[str] = Counter(
        name for node in model.graph.node for name in node.input if name
    )
    initializer_uses = {
        item.name: [
            {"node_index": index, "op_type": node.op_type, "input_index": list(node.input).index(item.name)}
            for index, node in enumerate(model.graph.node)
            if item.name in node.input
        ]
        for item in model.graph.initializer
    }
    unused_initializers = [name for name, uses in initializer_uses.items() if not uses]
    dead_node_outputs = [
        name for name in node_outputs if consumers[name] == 0 and name not in graph_outputs
    ]

    duplicate_initializers: list[list[str]] = []
    init_groups: defaultdict[tuple[int, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    initializers = []
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        key = (int(item.data_type), tuple(int(x) for x in item.dims), array.tobytes())
        init_groups[key].append(item.name)
        finite = bool(np.all(np.isfinite(array))) if array.dtype.kind in "fc" else True
        initializers.append(
            {
                "name": item.name,
                "dtype": TensorProto.DataType.Name(item.data_type),
                "shape": list(item.dims),
                "elements": int(array.size),
                "finite": finite,
                "zero_count": int(np.count_nonzero(array == 0)),
                "min": float(np.min(array)) if array.size else None,
                "max": float(np.max(array)) if array.size else None,
                "uses": initializer_uses[item.name],
            }
        )
    duplicate_initializers.extend(sorted(names) for names in init_groups.values() if len(names) > 1)

    duplicate_nodes: list[list[int]] = []
    node_groups: defaultdict[bytes, list[int]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        canonical = copy.deepcopy(node)
        canonical.name = ""
        del canonical.output[:]
        node_groups[canonical.SerializeToString(deterministic=True)].append(index)
    duplicate_nodes.extend(indices for indices in node_groups.values() if len(indices) > 1)

    identity_sites: list[dict[str, Any]] = []
    init_arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    elem_types = {
        name: int(value.type.tensor_type.elem_type) for name, value in typed.items()
    }
    for index, node in enumerate(model.graph.node):
        reason = None
        if node.op_type == "Add" and any(
            name in init_arrays and np.all(init_arrays[name] == 0) for name in node.input
        ):
            reason = "Add zero"
        elif node.op_type == "Mul" and any(
            name in init_arrays and np.all(init_arrays[name] == 1) for name in node.input
        ):
            reason = "Mul one"
        elif node.op_type in {"Sub", "Min"} and len(node.input) == 2 and node.input[0] == node.input[1]:
            reason = f"{node.op_type} identical inputs"
        elif node.op_type == "Where" and len(node.input) == 3 and node.input[1] == node.input[2]:
            reason = "Where identical branches"
        elif node.op_type == "Concat" and len(node.input) <= 1:
            reason = "single-input Concat"
        elif node.op_type == "Cast" and node.input and node.output:
            target = next((helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "to"), None)
            if target is not None and elem_types.get(node.input[0]) == int(target):
                reason = "same-type Cast"
        elif node.op_type == "BitwiseAnd" and any(
            name in init_arrays
            and init_arrays[name].dtype.kind == "u"
            and np.all(init_arrays[name] == np.iinfo(init_arrays[name].dtype).max)
            for name in node.input
        ):
            reason = "BitwiseAnd all-ones"
        if reason:
            identity_sites.append({"node_index": index, "op_type": node.op_type, "reason": reason})

    conv_bias_findings = []
    for index, node in enumerate(model.graph.node):
        if node.op_type == "Conv" and len(node.input) >= 3 and node.input[2]:
            bias = init_arrays.get(node.input[2])
            if bias is not None:
                conv_bias_findings.append(
                    {
                        "node_index": index,
                        "op_type": node.op_type,
                        "bias": node.input[2],
                        "dtype": str(bias.dtype),
                    }
                )
        if node.op_type == "QLinearConv" and len(node.input) >= 9 and node.input[8]:
            bias = init_arrays.get(node.input[8])
            conv_bias_findings.append(
                {
                    "node_index": index,
                    "op_type": node.op_type,
                    "bias": node.input[8],
                    "dtype": None if bias is None else str(bias.dtype),
                    "valid_int32": bool(bias is not None and bias.dtype == np.int32),
                }
            )

    standard_domains = all(node.domain in {"", "ai.onnx"} for node in model.graph.node)
    lookup_ops = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in {"CategoryMapper", "DictVectorizer", "LabelEncoder", "TfIdfVectorizer"}
        }
    )
    result = {
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "ir_version": int(model.ir_version),
        "opsets": [{"domain": item.domain, "version": int(item.version)} for item in model.opset_import],
        "graph_name": model.graph.name,
        "node_count": len(model.graph.node),
        "op_histogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "initializer_count": len(model.graph.initializer),
        "value_info_count": len(model.graph.value_info),
        "input": [value_record(value) for value in inferred.graph.input],
        "output": [value_record(value) for value in inferred.graph.output],
        "all_node_outputs_static_positive": all(row["static_positive"] for row in inferred_records),
        "inferred_intermediate_memory": memory,
        "params": params,
        "derived_cost": memory + params,
        "standard_domains": standard_domains,
        "nested_graph_count": nested_graph_count(model),
        "functions_count": len(model.functions),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "lookup_ops": lookup_ops,
        "finite_initializers": all(row["finite"] for row in initializers),
        "initializers": initializers,
        "unused_initializers": unused_initializers,
        "dead_node_outputs": dead_node_outputs,
        "duplicate_initializers": duplicate_initializers,
        "duplicate_node_groups": duplicate_nodes,
        "unconditional_identity_sites": identity_sites,
        "conv_bias_findings": conv_bias_findings,
        "conv_bias_ub0": all(
            item["op_type"] != "QLinearConv" or item.get("valid_int32", False)
            for item in conv_bias_findings
        ),
    }
    return model, inferred, result


def official_profile(model: onnx.ModelProto) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="task046_residual_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(model), TASK, workdir, label="task046_authority", require_correct=False
        )
    if result is None:
        raise RuntimeError("score_and_verify returned None")
    return result


def traced_model(model: onnx.ModelProto, inferred: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
    typed = {
        value.name: value
        for value in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
    }
    result = copy.deepcopy(model)
    existing = {value.name for value in result.graph.output}
    names: list[str] = []
    for node in result.graph.node:
        for name in node.output:
            if not name or name in names or name not in typed:
                continue
            names.append(name)
            if name not in existing:
                result.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
    return result, names


def make_session(model: onnx.ModelProto, disable_all: bool, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def nonfinite_signature(names: list[str], arrays: list[np.ndarray]) -> dict[str, dict[str, int]]:
    signature: dict[str, dict[str, int]] = {}
    for name, raw in zip(names, arrays):
        array = np.asarray(raw)
        if array.dtype.kind not in "fc":
            continue
        counts = {
            "nan": int(np.count_nonzero(np.isnan(array))),
            "posinf": int(np.count_nonzero(np.isposinf(array))),
            "neginf": int(np.count_nonzero(np.isneginf(array))),
        }
        if any(counts.values()):
            signature[name] = counts
    return signature


def scalar_token(array: np.ndarray) -> str:
    value = np.asarray(array).reshape(-1)[0]
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "+Inf"
        if np.isneginf(value):
            return "-Inf"
        return repr(float(value))
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    return str(int(value))


def case_digest_update(hasher: Any, array: np.ndarray) -> None:
    value = np.asarray(array)
    hasher.update(str(value.dtype).encode())
    hasher.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    hasher.update(value.tobytes(order="C"))


def run_suite(
    traced: onnx.ModelProto,
    trace_names: list[str],
    cases: list[dict[str, np.ndarray]],
    collect_values: bool,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    output_name = traced.graph.output[0].name
    input_name = traced.graph.input[0].name
    scalar_values: defaultdict[str, set[str]] = defaultdict(set)
    rows: dict[str, Any] = {}
    canonical_raw_digest = None
    canonical_nonfinite_digest = None

    for disable, threads, label in CONFIGS:
        row: dict[str, Any] = {
            "cases": len(cases),
            "right": 0,
            "wrong": 0,
            "runtime_errors": 0,
            "output_nonfinite": {"nan": 0, "posinf": 0, "neginf": 0},
            "intermediate_nonfinite_total": {"nan": 0, "posinf": 0, "neginf": 0},
            "intermediate_nonfinite_by_tensor": {},
            "cases_with_nonfinite_intermediate": 0,
            "nonfinite_signature_histogram": {},
            "output_shapes": [],
            "output_dtypes": [],
            "output_min": None,
            "output_max": None,
            "first_failure": None,
        }
        raw_hasher = hashlib.sha256()
        signatures: list[dict[str, dict[str, int]]] = []
        signature_histogram: Counter[str] = Counter()
        by_tensor: defaultdict[str, Counter[str]] = defaultdict(Counter)
        try:
            session = make_session(traced, disable, threads)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            row["runtime_errors"] = len(cases)
            rows[label] = row
            continue
        for case_index, benchmark in enumerate(cases):
            try:
                arrays = session.run(trace_names, {input_name: benchmark["input"]})
                values = {name: np.asarray(array) for name, array in zip(trace_names, arrays)}
                raw = values[output_name]
                case_digest_update(raw_hasher, raw)
                shape = list(raw.shape)
                dtype = str(raw.dtype)
                if shape not in row["output_shapes"]:
                    row["output_shapes"].append(shape)
                if dtype not in row["output_dtypes"]:
                    row["output_dtypes"].append(dtype)
                finite_output = raw[np.isfinite(raw)] if raw.dtype.kind in "fc" else raw.reshape(-1)
                if finite_output.size:
                    low = float(np.min(finite_output))
                    high = float(np.max(finite_output))
                    row["output_min"] = low if row["output_min"] is None else min(row["output_min"], low)
                    row["output_max"] = high if row["output_max"] is None else max(row["output_max"], high)
                if raw.dtype.kind in "fc":
                    row["output_nonfinite"]["nan"] += int(np.count_nonzero(np.isnan(raw)))
                    row["output_nonfinite"]["posinf"] += int(np.count_nonzero(np.isposinf(raw)))
                    row["output_nonfinite"]["neginf"] += int(np.count_nonzero(np.isneginf(raw)))
                expected = benchmark["output"].astype(bool)
                if np.array_equal(raw > 0, expected):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {
                            "case": case_index,
                            "different_cells": int(np.count_nonzero((raw > 0) != expected)),
                        }
                signature = nonfinite_signature(trace_names, arrays)
                signatures.append(signature)
                token = json.dumps(signature, sort_keys=True, separators=(",", ":"))
                signature_histogram[token] += 1
                if signature:
                    row["cases_with_nonfinite_intermediate"] += 1
                for name, counts in signature.items():
                    for kind, count in counts.items():
                        by_tensor[name][kind] += count
                        row["intermediate_nonfinite_total"][kind] += count
                if collect_values:
                    for name, value in values.items():
                        if value.size == 1 and len(scalar_values[name]) <= len(cases):
                            scalar_values[name].add(scalar_token(value))
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": case_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        row["raw_output_digest"] = raw_hasher.hexdigest()
        signature_bytes = json.dumps(signatures, sort_keys=True, separators=(",", ":")).encode()
        row["nonfinite_case_signature_digest"] = sha256(signature_bytes)
        row["nonfinite_signature_histogram"] = {
            token: count for token, count in sorted(signature_histogram.items())
        }
        row["intermediate_nonfinite_by_tensor"] = {
            name: dict(sorted(counts.items())) for name, counts in sorted(by_tensor.items())
        }
        row["perfect"] = (
            row["right"] == len(cases)
            and row["wrong"] == 0
            and row["runtime_errors"] == 0
            and not any(row["output_nonfinite"].values())
        )
        if canonical_raw_digest is None:
            canonical_raw_digest = row["raw_output_digest"]
            canonical_nonfinite_digest = row["nonfinite_case_signature_digest"]
        row["raw_output_matches_reference_config"] = row["raw_output_digest"] == canonical_raw_digest
        row["nonfinite_signature_matches_reference_config"] = (
            row["nonfinite_case_signature_digest"] == canonical_nonfinite_digest
        )
        rows[label] = row
    return rows, dict(scalar_values)


def known_cases() -> list[dict[str, np.ndarray]]:
    cases = []
    examples = scoring.load_examples(TASK)
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                raise RuntimeError(f"unconvertible known example in {split}")
            cases.append(benchmark)
    return cases


def fresh_cases() -> list[dict[str, np.ndarray]]:
    generator = importlib.import_module(f"task_{TASK_HASH}")
    random.seed(FRESH_SEED)
    np.random.seed(FRESH_SEED & 0xFFFFFFFF)
    cases = []
    attempts = 0
    while len(cases) < FRESH_COUNT:
        attempts += 1
        benchmark = scoring.convert_to_numpy(generator.generate())
        if benchmark is not None:
            cases.append(benchmark)
        if attempts > FRESH_COUNT * 10:
            raise RuntimeError("fresh generator failed to produce enough convertible cases")
    return cases


def merge_scalar_profiles(*profiles: dict[str, set[str]]) -> dict[str, Any]:
    merged: defaultdict[str, set[str]] = defaultdict(set)
    for profile in profiles:
        for name, values in profile.items():
            merged[name].update(values)
    records = {
        name: {"unique_count": len(values), "values": sorted(values)}
        for name, values in sorted(merged.items())
    }
    constants = {
        name: next(iter(values)) for name, values in sorted(merged.items()) if len(values) == 1
    }
    targets = [
        "a_inc", "b_inc", "c_inc", "b2", "b3", "s2", "s3", "s4",
        "c1", "c2", "c3", "c4", "d1", "d2", "d3", "sh2", "sh3", "sh4",
        "sm0", "xor_sm0", *[f"and2_sm{i}" for i in range(1, 16)],
    ]
    return {
        "scalar_outputs": records,
        "constant_scalar_outputs_on_known_plus_fresh": constants,
        "targeted_scalar_outputs": {name: records.get(name) for name in targets},
        "note": "Empirical ranges are diagnostic only and are not accepted as an all-input proof.",
    }


def main() -> int:
    zip_bytes = AUTHORITY_ZIP.read_bytes()
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        data = archive.read(MEMBER)
    member_sha = sha256(data)
    if member_sha != EXPECTED_MEMBER_SHA256:
        raise SystemExit(f"authority member drift: {member_sha}")

    model, inferred, structural = structural_audit(data)
    profile = official_profile(model)
    traced, trace_names = traced_model(model, inferred)
    known = known_cases()
    fresh = fresh_cases()
    known_four, known_values = run_suite(traced, trace_names, known, collect_values=True)
    fresh_four, fresh_values = run_suite(traced, trace_names, fresh, collect_values=True)

    output = {
        "authority": {
            "zip": AUTHORITY_ZIP.name,
            "zip_sha256": sha256(zip_bytes),
            "member": MEMBER,
            "member_sha256": member_sha,
            "serialized_bytes": len(data),
        },
        "generator": {
            "task_hash": TASK_HASH,
            "module": f"task_{TASK_HASH}",
            "support_summary": {
                "height": 3,
                "segment_count": [3, 4],
                "segment_width": [2, 4],
                "colors": "ARC colors excluding gray=5",
                "extra_input_column": [0, 1],
            },
            "fresh_seed": FRESH_SEED,
            "fresh_count": FRESH_COUNT,
        },
        "official_profile": profile,
        "structure": structural,
        "trace_output_count": len(trace_names),
        "known_count": len(known),
        "fresh_count": len(fresh),
        "known_four_configs": known_four,
        "fresh_four_configs": fresh_four,
        "observed_values": merge_scalar_profiles(known_values, fresh_values),
        "all_four_configs_known_perfect": all(row.get("perfect", False) for row in known_four.values()),
        "all_four_configs_fresh_perfect": all(row.get("perfect", False) for row in fresh_four.values()),
        "all_raw_output_digests_stable": all(
            row.get("raw_output_matches_reference_config", False)
            for row in [*known_four.values(), *fresh_four.values()]
        ),
        "all_nonfinite_signatures_stable": all(
            row.get("nonfinite_signature_matches_reference_config", False)
            for row in [*known_four.values(), *fresh_four.values()]
        ),
    }
    (HERE / "authority_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "member_sha256": member_sha,
                "official_cost": profile.get("cost"),
                "derived_cost": structural["derived_cost"],
                "known_count": len(known),
                "fresh_count": len(fresh),
                "known4": output["all_four_configs_known_perfect"],
                "fresh4": output["all_four_configs_fresh_perfect"],
                "raw_stable": output["all_raw_output_digests_stable"],
                "nonfinite_stable": output["all_nonfinite_signatures_stable"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
