#!/usr/bin/env python3
"""Systematic fail-closed Where-branch ablation scan over the 8009.46 authority."""

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxoptimizer
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, numpy_helper


ort.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATES_DIR = HERE / "candidates"
EVIDENCE = HERE / "evidence.json"
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
COST_CENSUS = ROOT / "scripts/golf/loop_8004_42_plus20/root_mem_census_119/canonical_costs.json"
ACTIVE_MANIFEST = ROOT / "others/71407/MANIFEST.json"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
EXPECTED_IO = (1, 10, 30, 30)
PRIORITY_COST_MIN = 150
PRIORITY_COST_MAX = 500
POLICY_THRESHOLD = 0.90
FRESH_PER_SEED = 2_000
GIANT_EINSUM_INPUTS = 15
GIANT_INITIALIZER_ELEMENTS = 100_000
MAX_FILE_BYTES = 1_440_000
CONFIGS = (
    ("disable_threads1", True, 1),
    ("disable_threads4", True, 4),
    ("default_threads1", False, 1),
    ("default_threads4", False, 4),
)
BASE_CONFIG = "disable_threads1"

PRIVATE_ZERO_OR_UNSOUND = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}
LOOKUP_OPS = {
    "TfIdfVectorizer", "Hardmax", "GatherND", "ScatterND", "CategoryMapper", "OneHot",
}
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import runtime_shape_trace  # noqa: E402
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return sha256(path.read_bytes())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def value_signature(value: onnx.ValueInfoProto) -> tuple[int, tuple[int | str | None, ...]]:
    tensor = value.type.tensor_type
    dims: list[int | str | None] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return int(tensor.elem_type), tuple(dims)


def signature_map(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int | str | None, ...]]]:
    values = {
        value.name: value_signature(value)
        for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    }
    for initializer in model.graph.initializer:
        values[initializer.name] = (
            int(initializer.data_type), tuple(int(dim) for dim in initializer.dims)
        )
    return values


def signature_json(signature: tuple[int, tuple[int | str | None, ...]] | None) -> dict[str, Any] | None:
    if signature is None:
        return None
    return {"dtype": TensorProto.DataType.Name(signature[0]), "shape": list(signature[1])}


def fully_static(signature: tuple[int, tuple[int | str | None, ...]] | None) -> bool:
    return bool(signature is not None and all(isinstance(dim, int) and dim > 0 for dim in signature[1]))


def nested_graph_count(model: onnx.ModelProto) -> int:
    total = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                total += 1
                pending.extend(attribute.g.node)
            elif attribute.type == AttributeProto.GRAPHS:
                total += len(attribute.graphs)
                for graph in attribute.graphs:
                    pending.extend(graph.node)
    return total


def branch_ablation(
    original: onnx.ModelProto, node_index: int, branch_index: int
) -> onnx.ModelProto:
    model = copy.deepcopy(original)
    target = model.graph.node[node_index]
    if target.op_type != "Where" or branch_index not in (1, 2):
        raise ValueError("invalid Where ablation target")
    old_output = target.output[0]
    replacement = target.input[branch_index]
    graph_outputs = {value.name for value in model.graph.output}
    if old_output in graph_outputs:
        identity = helper.make_node(
            "Identity", [replacement], [old_output],
            name=f"where_ablation_{node_index}_{'true' if branch_index == 1 else 'false'}",
        )
        model.graph.node[node_index].CopyFrom(identity)
    else:
        for index, node in enumerate(model.graph.node):
            if index == node_index:
                continue
            for input_index, name in enumerate(node.input):
                if name == old_output:
                    node.input[input_index] = replacement
        del model.graph.node[node_index]
        stale = [index for index, value in enumerate(model.graph.value_info) if value.name == old_output]
        for index in reversed(stale):
            del model.graph.value_info[index]
    model = onnxoptimizer.optimize(
        model, ["eliminate_deadend", "eliminate_unused_initializer"]
    )
    return model


def official_profile(task: int, model: onnx.ModelProto, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"where287_{task:03d}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            copy.deepcopy(model), task, work, label=label, require_correct=False
        )


def structural_audit(task: int, model: onnx.ModelProto, data: bytes) -> dict[str, Any]:
    full_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # noqa: BLE001
        full_error = f"{type(exc).__name__}: {exc}"
    inferred = None
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
    except Exception as exc:  # noqa: BLE001
        strict_error = f"{type(exc).__name__}: {exc}"
    typed = signature_map(inferred) if inferred is not None else {}
    missing_outputs = sorted({
        name for node in model.graph.node for name in node.output if name and name not in typed
    })
    nonstatic_outputs = sorted({
        name for node in model.graph.node for name in node.output
        if name and name in typed and not fully_static(typed[name])
    })
    input_signature = [value_signature(value) for value in model.graph.input]
    output_signature = [value_signature(value) for value in model.graph.output]
    canonical_io = bool(
        len(model.graph.input) == 1 and model.graph.input[0].name == "input"
        and input_signature == [(TensorProto.FLOAT, EXPECTED_IO)]
        and len(model.graph.output) == 1 and model.graph.output[0].name == "output"
        and len(output_signature) == 1 and output_signature[0][1] == EXPECTED_IO
        and output_signature[0][0] != TensorProto.UNDEFINED
    )
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    nonfinite = sorted(
        name for name, array in arrays.items()
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()
    )
    external = sorted(
        item.name for item in model.graph.initializer
        if item.data_location == TensorProto.EXTERNAL or item.external_data
    )
    ops = [node.op_type for node in model.graph.node]
    lookup = sorted(set(ops) & LOOKUP_OPS)
    banned = sorted({op for op in ops if op in BANNED_OPS or "Sequence" in op})
    domains = sorted({node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")})
    max_einsum_inputs = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0
    )
    giant_initializers = sorted(
        name for name, array in arrays.items() if array.size >= GIANT_INITIALIZER_ELEMENTS
    )
    conv_findings = [
        {"op": op, "bias": bias, "bias_len": bias_len, "out_channels": out_ch}
        for op, bias, bias_len, out_ch in check_conv_bias(copy.deepcopy(model))
    ]
    trace: dict[str, Any] = {}
    static_prepass = bool(
        full_error is None and strict_error is None
        and canonical_io
        and not missing_outputs and not nonstatic_outputs and not nonfinite and not external
        and not lookup and not banned and not domains and nested_graph_count(model) == 0
        and not model.functions and not model.graph.sparse_initializer
        and max_einsum_inputs < GIANT_EINSUM_INPUTS and not giant_initializers
        and not conv_findings and len(data) <= MAX_FILE_BYTES
    )
    if static_prepass:
        try:
            trace = runtime_shape_trace(task, copy.deepcopy(model))
            trace["shape_cloak_findings"] = len(trace.get("declared_actual_mismatches", []))
            trace["truthful"] = not trace.get("error") and trace["shape_cloak_findings"] == 0
        except Exception as exc:  # noqa: BLE001
            trace = {
                "truthful": False,
                "shape_cloak_findings": 1,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        trace = {"truthful": False, "shape_cloak_findings": None, "not_run": "static_prepass_failed"}
    passed = bool(static_prepass and trace.get("truthful") is True)
    reasons = []
    if full_error:
        reasons.append("full_checker")
    if strict_error:
        reasons.append("strict_shape")
    if not canonical_io:
        reasons.append("noncanonical_io")
    if missing_outputs or nonstatic_outputs:
        reasons.append("untyped_or_nonstatic_node_output")
    if lookup:
        reasons.append("lookup")
    if banned or domains or nested_graph_count(model) or model.functions or model.graph.sparse_initializer or external:
        reasons.append("nonstandard_or_banned_structure")
    if max_einsum_inputs >= GIANT_EINSUM_INPUTS or giant_initializers:
        reasons.append("giant")
    if nonfinite:
        reasons.append("nonfinite_initializer")
    if conv_findings:
        reasons.append("conv_bias_ub")
    if len(data) > MAX_FILE_BYTES:
        reasons.append("file_size")
    if static_prepass and trace.get("truthful") is not True:
        reasons.append("runtime_shape_cloak")
    return {
        "pass": passed,
        "reasons": sorted(set(reasons)),
        "full_check": full_error is None,
        "full_check_error": full_error,
        "strict_shape_data_prop": strict_error is None,
        "strict_shape_error": strict_error,
        "canonical_io": canonical_io,
        "input_signature": [signature_json(item) for item in input_signature],
        "output_signature": [signature_json(item) for item in output_signature],
        "missing_node_outputs": missing_outputs,
        "nonstatic_node_outputs": nonstatic_outputs,
        "op_histogram": dict(Counter(ops)),
        "nonstandard_domains": domains,
        "lookup_ops": lookup,
        "banned_ops": banned,
        "nested_graphs": nested_graph_count(model),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": external,
        "nonfinite_initializers": nonfinite,
        "initializer_elements": int(sum(array.size for array in arrays.values())),
        "largest_initializer_elements": int(max((array.size for array in arrays.values()), default=0)),
        "giant_initializers": giant_initializers,
        "max_einsum_inputs": max_einsum_inputs,
        "giant_einsum": max_einsum_inputs >= GIANT_EINSUM_INPUTS,
        "conv_bias_ub_findings": conv_findings,
        "runtime_intermediate_trace": trace,
        "file_bytes": len(data),
    }


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def sign_difference(left: bytes, right: bytes) -> int:
    xor = np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)
    )
    return int(np.unpackbits(xor).sum())


def evaluate_config(
    runtime: ort.InferenceSession,
    cases: list[dict[str, Any]],
    baseline_signs: list[bytes | None] | None,
) -> tuple[dict[str, Any], list[bytes | None]]:
    right = wrong = errors = 0
    nonfinite_cases = nonfinite_elements = shape_mismatches = small_positive = 0
    minimum_positive = math.inf
    maximum_nonpositive = -math.inf
    sign_mismatch_cases = sign_mismatch_cells = 0
    signs: list[bytes | None] = []
    sign_digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    first_wrong = first_error = first_shape = first_sign = None
    started = time.monotonic()
    for index, example in enumerate(cases):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"case {index} became unconvertible")
        expected = benchmark["output"] > 0
        try:
            raw = np.asarray(runtime.run(["output"], {"input": benchmark["input"]})[0])
        except Exception as exc:  # noqa: BLE001
            errors += 1
            signs.append(None)
            if first_error is None:
                first_error = {"case": index, "error": f"{type(exc).__name__}: {exc}"}
            continue
        if tuple(raw.shape) != EXPECTED_IO:
            shape_mismatches += 1
            signs.append(None)
            if first_shape is None:
                first_shape = {"case": index, "shape": list(raw.shape)}
            continue
        finite = np.isfinite(raw)
        bad = int(np.count_nonzero(~finite))
        nonfinite_cases += int(bad > 0)
        nonfinite_elements += bad
        positive = raw > 0
        packed = np.packbits(positive.reshape(-1), bitorder="little").tobytes()
        signs.append(packed)
        sign_digest.update(packed)
        raw_digest.update(np.ascontiguousarray(raw).tobytes())
        correct = bool(np.array_equal(positive, expected))
        right += int(correct)
        wrong += int(not correct)
        if not correct and first_wrong is None:
            first_wrong = {"case": index, "different_cells": int(np.count_nonzero(positive != expected))}
        small_positive += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        if np.any(positive):
            minimum_positive = min(minimum_positive, float(raw[positive].min()))
        nonpositive = finite & ~positive
        if np.any(nonpositive):
            maximum_nonpositive = max(maximum_nonpositive, float(raw[nonpositive].max()))
        if baseline_signs is not None:
            baseline = baseline_signs[index]
            difference = math.prod(EXPECTED_IO) if baseline is None else sign_difference(packed, baseline)
            sign_mismatch_cases += int(difference > 0)
            sign_mismatch_cells += difference
            if difference and first_sign is None:
                first_sign = {"case": index, "different_cells": difference}
    total = len(cases)
    return ({
        "total": total,
        "right": right,
        "wrong": wrong,
        "accuracy": right / total,
        "policy90": right / total >= POLICY_THRESHOLD,
        "errors": errors,
        "nonfinite_cases": nonfinite_cases,
        "nonfinite_elements": nonfinite_elements,
        "runtime_shape_mismatches": shape_mismatches,
        "small_positive_elements_0_to_0_25": small_positive,
        "minimum_positive": None if minimum_positive == math.inf else minimum_positive,
        "maximum_nonpositive": None if maximum_nonpositive == -math.inf else maximum_nonpositive,
        "sign_mismatch_cases_vs_disable_threads1": sign_mismatch_cases,
        "sign_mismatch_cells_vs_disable_threads1": sign_mismatch_cells,
        "sign_sha256": sign_digest.hexdigest(),
        "raw_sha256": raw_digest.hexdigest(),
        "first_wrong": first_wrong,
        "first_error": first_error,
        "first_shape_mismatch": first_shape,
        "first_sign_mismatch": first_sign,
        "elapsed_seconds": time.monotonic() - started,
    }, signs)


def evaluate_four(data: bytes, cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    baseline_signs = None
    for name, disable_all, threads in CONFIGS:
        try:
            runtime = make_session(data, disable_all, threads)
            row, signs = evaluate_config(
                runtime, cases, None if name == BASE_CONFIG else baseline_signs
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "total": len(cases), "right": 0, "wrong": 0, "accuracy": 0.0,
                "policy90": False, "errors": len(cases),
                "session_error": f"{type(exc).__name__}: {exc}",
                "nonfinite_cases": 0, "nonfinite_elements": 0,
                "runtime_shape_mismatches": 0,
                "small_positive_elements_0_to_0_25": 0,
                "sign_mismatch_cases_vs_disable_threads1": 0,
                "sign_mismatch_cells_vs_disable_threads1": 0,
            }
            signs = [None] * len(cases)
        row["optimization"] = "ORT_DISABLE_ALL" if disable_all else "ORT_ENABLE_ALL"
        row["threads"] = threads
        rows[name] = row
        if name == BASE_CONFIG:
            baseline_signs = signs
    return rows


def runtime_row_pass(row: dict[str, Any]) -> bool:
    return bool(
        row.get("policy90") is True and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0 and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def four_pass(rows: dict[str, Any]) -> bool:
    return bool(len(rows) == 4 and all(runtime_row_pass(row) for row in rows.values()))


def known_cases(task: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = scoring.load_examples(task)
    cases = []
    raw = {}
    converted = {}
    for subset in ("train", "test", "arc-gen"):
        raw[subset] = len(examples[subset])
        converted[subset] = 0
        for example in examples[subset]:
            if scoring.convert_to_numpy(example) is None:
                continue
            converted[subset] += 1
            cases.append(example)
    return cases, {
        "raw": raw,
        "converted": converted,
        "raw_total": sum(raw.values()),
        "converted_total": sum(converted.values()),
        "all_cases_convertible": sum(raw.values()) == sum(converted.values()),
    }


def case_id(example: dict[str, Any]) -> str:
    digestor = hashlib.sha256()
    for key in ("input", "output"):
        array = np.asarray(example[key], dtype=np.uint8)
        digestor.update(np.asarray(array.shape, dtype=np.int16).tobytes())
        digestor.update(array.tobytes())
    return digestor.hexdigest()


def fresh_cases(task: int, seed: int, task_map: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    module_name = f"task_{task_map[f'{task:03d}']}"
    generator = importlib.import_module(module_name)
    common = importlib.import_module("common")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    common.random.seed(seed)
    cases = []
    attempts = generation_errors = conversion_skips = 0
    seen = set()
    stream = hashlib.sha256()
    while len(cases) < FRESH_PER_SEED:
        attempts += 1
        try:
            example = generator.generate()
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        if scoring.convert_to_numpy(example) is None:
            conversion_skips += 1
            continue
        identifier = case_id(example)
        seen.add(identifier)
        stream.update(bytes.fromhex(identifier))
        cases.append(example)
    return cases, {
        "task": task,
        "seed": seed,
        "module": module_name,
        "requested": FRESH_PER_SEED,
        "accepted": len(cases),
        "attempts": attempts,
        "generation_errors": generation_errors,
        "conversion_skips": conversion_skips,
        "unique_case_ids": len(seen),
        "case_stream_sha256": stream.hexdigest(),
    }


def main() -> int:
    started = time.monotonic()
    if digest(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority changed")
    census = json.loads(COST_CENSUS.read_text(encoding="utf-8"))
    authority_costs = {int(row["task"]): int(row["cost"]) for row in census["ranked"]}
    if len(authority_costs) != 400:
        raise RuntimeError("authority cost census is not complete")
    active_manifest = json.loads(ACTIVE_MANIFEST.read_text(encoding="utf-8"))
    active_tasks = {int(row["task"]) for row in active_manifest["active_candidates"]}
    if len(active_tasks) < 22:
        raise RuntimeError(f"active manifest regressed below assigned active22: {len(active_tasks)}")
    excluded_tasks = active_tasks | PRIVATE_ZERO_OR_UNSOUND
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    if len(task_map) != 400:
        raise RuntimeError("task hash map is not complete")
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    authority_data: dict[int, bytes] = {}
    authority_models: dict[int, onnx.ModelProto] = {}
    authority_inventory = []
    variants: list[dict[str, Any]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            data = archive.read(f"task{task:03d}.onnx")
            authority_data[task] = data
            cost = authority_costs[task]
            exclusions = []
            if task in active_tasks:
                exclusions.append("71407_active")
            if task in PRIVATE_ZERO_OR_UNSOUND:
                exclusions.append("private_zero_or_unsound_monitor")
            record: dict[str, Any] = {
                "task": task,
                "cost": cost,
                "sha256": sha256(data),
                "priority_cost_150_to_500": PRIORITY_COST_MIN <= cost <= PRIORITY_COST_MAX,
                "excluded": bool(exclusions),
                "exclusion_reasons": exclusions,
                "where_nodes": 0,
                "shape_exact_branch_variants": 0,
            }
            if exclusions:
                authority_inventory.append(record)
                continue
            model = onnx.load_model_from_string(data)
            authority_models[task] = model
            try:
                inferred = onnx.shape_inference.infer_shapes(
                    copy.deepcopy(model), strict_mode=True, data_prop=True
                )
                values = signature_map(inferred)
                record["strict_shape_data_prop"] = True
            except Exception as exc:  # noqa: BLE001
                record["strict_shape_data_prop"] = False
                record["strict_shape_error"] = f"{type(exc).__name__}: {exc}"
                authority_inventory.append(record)
                continue
            branch_rows = []
            for node_index, node in enumerate(inferred.graph.node):
                if node.op_type != "Where":
                    continue
                record["where_nodes"] += 1
                output_signature = values.get(node.output[0])
                for branch_index, label in ((1, "true"), (2, "false")):
                    branch_signature = values.get(node.input[branch_index])
                    exact = bool(
                        output_signature is not None
                        and branch_signature == output_signature
                        and fully_static(output_signature)
                    )
                    branch_rows.append({
                        "node_index": node_index,
                        "node_name": node.name,
                        "where_output": node.output[0],
                        "condition": node.input[0],
                        "branch": label,
                        "branch_index": branch_index,
                        "branch_value": node.input[branch_index],
                        "branch_signature": signature_json(branch_signature),
                        "output_signature": signature_json(output_signature),
                        "exact_shape_dtype_match": exact,
                    })
                    if exact:
                        record["shape_exact_branch_variants"] += 1
                        variants.append({
                            "task": task,
                            "authority_cost": cost,
                            "authority_sha256": sha256(data),
                            "priority_cost_150_to_500": PRIORITY_COST_MIN <= cost <= PRIORITY_COST_MAX,
                            **branch_rows[-1],
                        })
            record["branches"] = branch_rows
            authority_inventory.append(record)

    variants.sort(key=lambda row: (
        not row["priority_cost_150_to_500"], row["authority_cost"], row["task"],
        row["node_index"], row["branch_index"],
    ))
    print(json.dumps({
        "eligible_authorities": 400 - len(excluded_tasks),
        "authorities_with_where": sum(item["where_nodes"] > 0 for item in authority_inventory),
        "shape_exact_variants": len(variants),
        "priority_variants": sum(item["priority_cost_150_to_500"] for item in variants),
    }), flush=True)

    candidate_rows: list[dict[str, Any]] = []
    candidate_data: dict[str, bytes] = {}
    dedupe: dict[tuple[int, str], int] = {}
    for variant_index, variant in enumerate(variants, start=1):
        task = int(variant["task"])
        row = copy.deepcopy(variant)
        original = authority_models[task]
        try:
            candidate = branch_ablation(original, row["node_index"], row["branch_index"])
            data = candidate.SerializeToString()
            row["candidate_sha256"] = sha256(data)
            row["candidate_file_bytes"] = len(data)
            row["authority_node_count"] = len(original.graph.node)
            row["candidate_node_count"] = len(candidate.graph.node)
            row["authority_initializer_count"] = len(original.graph.initializer)
            row["candidate_initializer_count"] = len(candidate.graph.initializer)
        except Exception as exc:  # noqa: BLE001
            row["classification"] = "REJECT_BUILD"
            row["build_error"] = f"{type(exc).__name__}: {exc}"
            candidate_rows.append(row)
            continue
        key = (task, row["candidate_sha256"])
        if key in dedupe:
            row["classification"] = "REJECT_DUPLICATE_CANDIDATE"
            row["duplicate_of_candidate_index"] = dedupe[key]
            candidate_rows.append(row)
            continue
        dedupe[key] = len(candidate_rows)
        row["structure"] = structural_audit(task, candidate, data)
        if not row["structure"]["pass"]:
            row["classification"] = "REJECT_STRUCTURE"
            candidate_rows.append(row)
            continue
        profile = official_profile(
            task, candidate, f"w{row['node_index']}_{row['branch'][0]}"
        )
        row["official_profile"] = profile
        if profile is None:
            row["classification"] = "REJECT_UNSCORABLE"
            candidate_rows.append(row)
            continue
        row["cost_reduction"] = row["authority_cost"] - int(profile["cost"])
        row["projected_gain"] = (
            math.log(row["authority_cost"] / int(profile["cost"]))
            if int(profile["cost"]) > 0 else None
        )
        if int(profile["cost"]) >= row["authority_cost"]:
            row["classification"] = "REJECT_NOT_STRICT_LOWER_ACTUAL_COST"
            candidate_rows.append(row)
            continue
        row["classification"] = "QUALIFIED_STRICT_LOWER_STRUCTURE"
        candidate_data[row["candidate_sha256"]] = data
        candidate_rows.append(row)
        if variant_index % 25 == 0 or row["classification"] == "QUALIFIED_STRICT_LOWER_STRUCTURE":
            print(json.dumps({
                "variant": variant_index, "total": len(variants), "task": task,
                "branch": row["branch"], "class": row["classification"],
                "cost": profile["cost"], "authority_cost": row["authority_cost"],
            }), flush=True)

    strict_rows = [row for row in candidate_rows if row["classification"] == "QUALIFIED_STRICT_LOWER_STRUCTURE"]
    known_cache: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for index, row in enumerate(strict_rows, start=1):
        task = int(row["task"])
        if task not in known_cache:
            known_cache[task] = known_cases(task)
        cases, counts = known_cache[task]
        row["known_counts"] = counts
        row["known_four"] = evaluate_four(candidate_data[row["candidate_sha256"]], cases)
        row["known_pass"] = bool(counts["all_cases_convertible"] and four_pass(row["known_four"]))
        row["classification"] = (
            "PASS_KNOWN_FOUR" if row["known_pass"] else "REJECT_KNOWN_FOUR"
        )
        print(json.dumps({
            "known": index, "total": len(strict_rows), "task": task,
            "cost": row["official_profile"]["cost"], "pass": row["known_pass"],
            "accuracy": {name: item["accuracy"] for name, item in row["known_four"].items()},
        }), flush=True)

    known_pass_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        if row.get("known_pass"):
            known_pass_by_task[int(row["task"])].append(row)
    for rows in known_pass_by_task.values():
        rows.sort(key=lambda row: (
            int(row["official_profile"]["cost"]), row["candidate_sha256"],
        ))

    fresh_cache: dict[int, list[tuple[list[dict[str, Any]], dict[str, Any]]]] = {}
    reported = []
    fresh_audits = []
    for task in sorted(known_pass_by_task):
        seeds = (287_000_000 + task, 287_100_000 + task)
        fresh_cache[task] = [fresh_cases(task, seed, task_map) for seed in seeds]
        task_pass = None
        for candidate_index, row in enumerate(known_pass_by_task[task]):
            seed_rows = []
            for cases, generation in fresh_cache[task]:
                four = evaluate_four(candidate_data[row["candidate_sha256"]], cases)
                seed_rows.append({"generation": generation, "runtime": four, "pass": four_pass(four)})
                print(json.dumps({
                    "fresh_task": task, "seed": generation["seed"],
                    "candidate_cost": row["official_profile"]["cost"],
                    "pass": four_pass(four),
                    "accuracy": {name: item["accuracy"] for name, item in four.items()},
                }), flush=True)
            passed = all(item["pass"] for item in seed_rows)
            audit = {
                "task": task,
                "candidate_sha256": row["candidate_sha256"],
                "candidate_cost": row["official_profile"]["cost"],
                "authority_cost": row["authority_cost"],
                "fresh_pass": passed,
                "seeds": list(seeds),
                "count_per_seed": FRESH_PER_SEED,
                "runs": seed_rows,
            }
            fresh_audits.append(audit)
            row["fresh_audit"] = audit
            row["fresh_pass"] = passed
            row["classification"] = "PASS_FRESH_FOUR" if passed else "REJECT_FRESH_FOUR"
            if passed:
                task_pass = row
                for skipped in known_pass_by_task[task][candidate_index + 1:]:
                    skipped["classification"] = "NOT_FRESH_TESTED_MORE_EXPENSIVE_AFTER_TASK_PASS"
                break
        if task_pass is not None:
            filename = (
                f"task{task:03d}_where{task_pass['node_index']:03d}_"
                f"{task_pass['branch']}_cost{task_pass['official_profile']['cost']}.onnx"
            )
            path = CANDIDATES_DIR / filename
            path.write_bytes(candidate_data[task_pass["candidate_sha256"]])
            if digest(path) != task_pass["candidate_sha256"]:
                raise RuntimeError("saved candidate SHA mismatch")
            task_pass["saved_path"] = rel(path)
            reported.append({
                "task": task,
                "path": rel(path),
                "sha256": task_pass["candidate_sha256"],
                "cost": task_pass["official_profile"]["cost"],
                "authority_cost": task_pass["authority_cost"],
                "cost_reduction": task_pass["cost_reduction"],
                "projected_gain": task_pass["projected_gain"],
                "where_node_index": task_pass["node_index"],
                "where_node_name": task_pass["node_name"],
                "branch": task_pass["branch"],
                "branch_value": task_pass["branch_value"],
                "candidate_row_sha256": task_pass["candidate_sha256"],
            })

    for row in candidate_rows:
        row.pop("_data", None)
    classifications = Counter(row["classification"] for row in candidate_rows)
    payload = {
        "lane": "agent_where_ablation_scan_287",
        "decision": "PASS_CANDIDATES_FOUND" if reported else "NO_PASSING_WHERE_ABLATION",
        "authority": {
            "zip": rel(AUTHORITY),
            "sha256": AUTHORITY_SHA256,
            "task_count": 400,
            "cost_census": rel(COST_CENSUS),
            "cost_census_sha256": digest(COST_CENSUS),
        },
        "policy": {
            "threshold": POLICY_THRESHOLD,
            "priority_cost_range": [PRIORITY_COST_MIN, PRIORITY_COST_MAX],
            "fresh_per_seed": FRESH_PER_SEED,
            "fresh_seed_formula": ["287000000 + task", "287100000 + task"],
            "configs": [
                {"name": name, "optimization": "ORT_DISABLE_ALL" if disable else "ORT_ENABLE_ALL", "threads": threads}
                for name, disable, threads in CONFIGS
            ],
            "shape_dtype_exact_branch_only": True,
            "onnxoptimizer_passes": ["eliminate_deadend", "eliminate_unused_initializer"],
            "strict_lower_actual_cost_only": True,
            "assigned_active22_and_any_newer_active_excluded": True,
            "private_zero_excluded": True,
            "lookup_shape_cloak_giant_excluded": True,
            "root_submission_scores_others71407_written": False,
            "kimi_used": False,
        },
        "exclusions": {
            "active_manifest": rel(ACTIVE_MANIFEST),
            "active_manifest_sha256": digest(ACTIVE_MANIFEST),
            "assigned_active_count": 22,
            "observed_active_count": len(active_tasks),
            "active_tasks": sorted(active_tasks),
            "private_zero_or_unsound_monitor": sorted(PRIVATE_ZERO_OR_UNSOUND),
            "excluded_union_count": len(excluded_tasks),
        },
        "coverage": {
            "authority_tasks": 400,
            "eligible_authority_tasks": 400 - len(excluded_tasks),
            "priority_authority_tasks": sum(
                item["priority_cost_150_to_500"] and not item["excluded"] for item in authority_inventory
            ),
            "authorities_with_where": sum(item["where_nodes"] > 0 for item in authority_inventory),
            "where_nodes": sum(item["where_nodes"] for item in authority_inventory),
            "shape_exact_branch_variants": len(variants),
            "priority_shape_exact_branch_variants": sum(item["priority_cost_150_to_500"] for item in variants),
            "deduplicated_candidate_count": len(dedupe),
            "strict_lower_structure_count": len(strict_rows),
            "known_four_pass_count": sum(bool(row.get("known_pass")) for row in strict_rows),
            "known_four_pass_tasks": sorted(known_pass_by_task),
            "fresh_audited_candidate_count": len(fresh_audits),
            "reported_task_count": len(reported),
            "reported_tasks": [row["task"] for row in reported],
            "one_cheapest_report_per_task": True,
        },
        "classification_counts": dict(classifications),
        "authority_inventory": authority_inventory,
        "candidate_rows": candidate_rows,
        "fresh_audits": fresh_audits,
        "reported_candidates": reported,
        "aggregate": {
            "known_case_config_executions": int(sum(
                item["total"] for row in strict_rows for item in row.get("known_four", {}).values()
            )),
            "fresh_case_config_executions": int(sum(
                item["total"] for audit in fresh_audits for run in audit["runs"]
                for item in run["runtime"].values()
            )),
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "coverage": payload["coverage"],
        "classifications": payload["classification_counts"],
        "reported": reported,
        "elapsed_seconds": payload["aggregate"]["elapsed_seconds"],
        "evidence": rel(EVIDENCE),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
