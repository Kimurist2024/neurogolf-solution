#!/usr/bin/env python3
"""Fail-closed SparseTensorProto experiments for the staged task192 model.

The experiment varies both legal COO encodings and the type metadata exposed
to Einsum.  No candidate is eligible unless the full checker, strict shape
inference with data propagation, raw and sanitized ORT sessions, and the
competition profiler all accept it.  A Constant(sparse_value=...) control is
included to distinguish malformed COO storage from an unsupported sparse
initializer consumer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = REPO / "others" / "71407" / "task192.onnx"
CANDIDATES = HERE / "candidates"
WORK = HERE / "work"
TASK = 192
ADJ = "adj"
ZERO = np.zeros((1, 10, 30, 30), dtype=np.float32)

sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def attempt(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = fn()
        return {"ok": True, "result": value}
    except BaseException as exc:  # Record every independent gate, fail closed.
        return {"ok": False, "error": error_text(exc)}


def source_adj(model: onnx.ModelProto) -> tuple[np.ndarray, onnx.TensorProto]:
    tensor = next(item for item in model.graph.initializer if item.name == ADJ)
    return np.asarray(numpy_helper.to_array(tensor)), tensor


def coo(array: np.ndarray, form: str, values_name: str) -> onnx.SparseTensorProto:
    flat = array.reshape(-1)
    linear = np.flatnonzero(flat != 0).astype(np.int64)
    values = np.asarray(flat[linear], dtype=array.dtype)
    if form == "linear":
        indices = linear
    elif form == "coordinates":
        indices = np.column_stack(np.unravel_index(linear, array.shape)).astype(np.int64)
    else:
        raise ValueError(form)
    return helper.make_sparse_tensor(
        numpy_helper.from_array(values, name=values_name),
        numpy_helper.from_array(indices, name=""),
        list(array.shape),
    )


def dense_from_sparse(sparse: onnx.SparseTensorProto) -> np.ndarray:
    values = np.asarray(numpy_helper.to_array(sparse.values))
    indices = np.asarray(numpy_helper.to_array(sparse.indices), dtype=np.int64)
    dense = np.zeros(tuple(sparse.dims), dtype=values.dtype)
    if indices.ndim == 1:
        dense.reshape(-1)[indices] = values.reshape(-1)
    elif indices.ndim == 2 and indices.shape[1] == dense.ndim:
        dense[tuple(indices.T)] = values.reshape(-1)
    else:
        raise ValueError(f"invalid COO indices shape {indices.shape}")
    return dense


def replace_name(model: onnx.ModelProto, old: str, new: str) -> None:
    for item in model.graph.input:
        if item.name == old:
            item.name = new
    for item in model.graph.value_info:
        if item.name == old:
            item.name = new
    for item in model.graph.output:
        if item.name == old:
            item.name = new
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new


def mapped_placeholder(model: onnx.ModelProto, placeholder: str) -> str:
    """Find the sanitizer name assigned to a placeholder graph reference."""
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitizer rejected probe model")
    original_refs: list[tuple[int, int]] = []
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name == placeholder:
                original_refs.append((node_index, input_index))
    if original_refs:
        node_index, input_index = original_refs[0]
        return sanitized.graph.node[node_index].input[input_index]
    for index, item in enumerate(model.graph.input):
        if item.name == placeholder:
            return sanitized.graph.input[index].name
    for index, item in enumerate(model.graph.value_info):
        if item.name == placeholder:
            return sanitized.graph.value_info[index].name
    raise RuntimeError("placeholder has no graph reference")


def add_metadata(model: onnx.ModelProto, placeholder: str, mode: str) -> None:
    if mode == "none":
        return
    if mode == "dense_value_info":
        model.graph.value_info.append(
            helper.make_tensor_value_info(placeholder, TensorProto.FLOAT, [30, 30])
        )
        return
    if mode == "sparse_value_info":
        model.graph.value_info.append(
            helper.make_sparse_tensor_value_info(
                placeholder, TensorProto.FLOAT, [30, 30]
            )
        )
        return
    if mode == "dense_graph_input":
        model.graph.input.append(
            helper.make_tensor_value_info(placeholder, TensorProto.FLOAT, [30, 30])
        )
        return
    if mode == "sparse_graph_input":
        model.graph.input.append(
            helper.make_sparse_tensor_value_info(
                placeholder, TensorProto.FLOAT, [30, 30]
            )
        )
        return
    raise ValueError(mode)


def build_sparse_initializer(
    source: onnx.ModelProto, form: str, metadata: str
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(source)
    array, _ = source_adj(model)
    placeholder = "__adj_sparse_placeholder__"
    kept = [item for item in model.graph.initializer if item.name != ADJ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    replace_name(model, ADJ, placeholder)
    add_metadata(model, placeholder, metadata)
    sparse = coo(array, form, placeholder)
    model.graph.sparse_initializer.append(sparse)

    # scoring.sanitize_model renames graph references but not SparseTensorProto
    # values.name.  Make this identifier a sanitizer fixed point so the test is
    # about sparse type support, not an accidental missing initializer binding.
    fixed_name = mapped_placeholder(model, placeholder)
    replace_name(model, placeholder, fixed_name)
    model.graph.sparse_initializer[-1].values.name = fixed_name

    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitizer rejected fixed-point model")
    consumer_names = {
        name
        for node in sanitized.graph.node
        if node.op_type == "Einsum"
        for name in node.input
    }
    binding_ok = fixed_name in consumer_names and any(
        item.values.name == fixed_name for item in sanitized.graph.sparse_initializer
    )
    rebuilt = dense_from_sparse(model.graph.sparse_initializer[-1])
    exact = bool(np.array_equal(array, rebuilt, equal_nan=True))
    if not exact or not binding_ok:
        raise AssertionError(f"exact={exact}, sanitizer_binding={binding_ok}")
    return model, {
        "fixed_name": fixed_name,
        "sanitizer_binding": binding_ok,
        "dense_reconstruction": "BIT_IDENTICAL",
        "indices_shape": list(model.graph.sparse_initializer[-1].indices.dims),
        "values": int(np.count_nonzero(array)),
    }


def build_sparse_constant(source: onnx.ModelProto, form: str) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(source)
    array, _ = source_adj(model)
    kept = [item for item in model.graph.initializer if item.name != ADJ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    sparse = coo(array, form, "")
    constant = helper.make_node("Constant", [], [ADJ], sparse_value=sparse, name="adj_sparse")
    model.graph.node.insert(0, constant)
    rebuilt = dense_from_sparse(sparse)
    if not np.array_equal(array, rebuilt, equal_nan=True):
        raise AssertionError("sparse Constant reconstruction differs")
    return model, {
        "dense_reconstruction": "BIT_IDENTICAL",
        "indices_shape": list(sparse.indices.dims),
        "values": int(np.count_nonzero(array)),
    }


def checker(model: onnx.ModelProto, full: bool) -> str:
    onnx.checker.check_model(model, full_check=full)
    return "PASS"


def strict_inference(model: onnx.ModelProto) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    types: dict[str, str] = {}
    for item in list(inferred.graph.input) + list(inferred.graph.value_info):
        if item.name in {s.values.name for s in inferred.graph.sparse_initializer}:
            types[item.name] = str(item.type).strip()
    return {"pass": True, "sparse_binding_types": types}


def input_array() -> np.ndarray:
    examples = scoring.load_examples(TASK)
    for subset in ("train", "test", "arc-gen"):
        for example in examples.get(subset, []):
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                return converted["input"]
    return ZERO


def session_options(level: str) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if level == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif level == "default":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        raise ValueError(level)
    return options


def ort_run(
    model: onnx.ModelProto,
    level: str,
    sanitize: bool,
    benchmark_input: np.ndarray,
    adj: np.ndarray,
) -> dict[str, Any]:
    tested = copy.deepcopy(model)
    if sanitize:
        tested = scoring.sanitize_model(tested)
        if tested is None:
            raise RuntimeError("sanitizer rejected model")
    session = ort.InferenceSession(
        tested.SerializeToString(), session_options(level), providers=["CPUExecutionProvider"]
    )
    feed: dict[str, np.ndarray] = {}
    exposed: list[dict[str, Any]] = []
    for item in session.get_inputs():
        exposed.append({"name": item.name, "shape": item.shape, "type": item.type})
        if item.name == "input":
            feed[item.name] = benchmark_input
        elif item.shape == [30, 30] and "float" in item.type:
            feed[item.name] = adj
        else:
            raise RuntimeError(f"cannot feed exposed input {item.name}: {item.shape} {item.type}")
    output = session.run(["output"], feed)[0]
    return {
        "exposed_inputs": exposed,
        "output_shape": list(output.shape),
        "finite": int(np.isfinite(output).sum()),
        "elements": int(output.size),
        "nonfinite": int(output.size - np.isfinite(output).sum()),
        "threshold_true": int(np.count_nonzero(output > 0.0)),
        "raw_sha256": hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest(),
    }


def competition_profile(model: onnx.ModelProto, label: str) -> Any:
    result = scoring.score_and_verify(
        copy.deepcopy(model), TASK, str(WORK), label=label, require_correct=False
    )
    if result is None:
        raise RuntimeError("official-compatible score_and_verify rejected candidate")
    return result


def cost_profile(model: onnx.ModelProto, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"sparse173_{label}_") as directory:
        path = Path(directory) / "task192.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    if cost < 0:
        raise RuntimeError(f"cost_of rejected candidate: {(memory, params, cost)}")
    return {"memory": memory, "params": params, "cost": cost}


def schema_evidence() -> dict[str, Any]:
    schema = onnx.defs.get_schema("Einsum", 18, "")
    allowed = sorted(str(item) for item in schema.type_constraints[0].allowed_type_strs)
    return {
        "domain": schema.domain,
        "since_version": schema.since_version,
        "type_parameter": schema.type_constraints[0].type_param_str,
        "allowed_types": allowed,
        "allows_sparse_tensor": any("sparse_tensor" in item for item in allowed),
    }


def run_variant(
    source: onnx.ModelProto,
    source_output: dict[str, dict[str, Any]],
    form: str,
    metadata: str,
) -> dict[str, Any]:
    label = f"initializer_{form}_{metadata}"
    row: dict[str, Any] = {
        "label": label,
        "kind": "sparse_initializer",
        "indices": form,
        "metadata": metadata,
    }
    built = attempt(lambda: build_sparse_initializer(source, form, metadata))
    row["build"] = built
    if not built["ok"]:
        row.update({"eligible": False, "decision": "REJECT_BUILD"})
        return row
    model, build_info = built["result"]
    built["result"] = build_info
    path = CANDIDATES / f"task192_{label}.onnx"
    onnx.save(model, path)
    row.update({"path": str(path.relative_to(REPO)), "sha256": sha256(path)})
    row["checker_basic"] = attempt(lambda: checker(model, False))
    row["checker_full"] = attempt(lambda: checker(model, True))
    row["strict_data_prop"] = attempt(lambda: strict_inference(model))
    benchmark_input = input_array()
    adj, _ = source_adj(source)
    for sanitize in (False, True):
        for level in ("default", "disabled"):
            key = f"ort_{'sanitized' if sanitize else 'raw'}_{level}"
            row[key] = attempt(
                lambda level=level, sanitize=sanitize: ort_run(
                    model, level, sanitize, benchmark_input, adj
                )
            )
            if row[key]["ok"]:
                reference = source_output[level]["raw_sha256"]
                row[key]["result"]["source_raw_sha256"] = reference
                row[key]["result"]["bit_identical_to_source"] = (
                    row[key]["result"]["raw_sha256"] == reference
                )
    row["cost_profile"] = attempt(lambda: cost_profile(model, label))
    row["competition_profile"] = attempt(lambda: competition_profile(model, label))
    mandatory = [
        "checker_basic",
        "checker_full",
        "strict_data_prop",
        "ort_raw_default",
        "ort_raw_disabled",
        "ort_sanitized_default",
        "ort_sanitized_disabled",
        "cost_profile",
        "competition_profile",
    ]
    row["eligible"] = all(row[item]["ok"] for item in mandatory)
    if row["eligible"]:
        cp = row["competition_profile"]["result"]
        row["eligible"] = bool(cp["cost"] < 1195)
    row["decision"] = "ELIGIBLE_FOR_DEEP_AUDIT" if row["eligible"] else "REJECT"
    return row


def run_control(
    source: onnx.ModelProto,
    source_output: dict[str, dict[str, Any]],
    form: str,
) -> dict[str, Any]:
    label = f"constant_{form}"
    model, build_info = build_sparse_constant(source, form)
    path = CANDIDATES / f"task192_{label}.onnx"
    onnx.save(model, path)
    row: dict[str, Any] = {
        "label": label,
        "kind": "constant_sparse_value_control",
        "indices": form,
        "metadata": "not_applicable",
        "build": {"ok": True, "result": build_info},
        "path": str(path.relative_to(REPO)),
        "sha256": sha256(path),
    }
    row["checker_basic"] = attempt(lambda: checker(model, False))
    row["checker_full"] = attempt(lambda: checker(model, True))
    row["strict_data_prop"] = attempt(lambda: strict_inference(model))
    benchmark_input = input_array()
    adj, _ = source_adj(source)
    for sanitize in (False, True):
        for level in ("default", "disabled"):
            key = f"ort_{'sanitized' if sanitize else 'raw'}_{level}"
            row[key] = attempt(
                lambda level=level, sanitize=sanitize: ort_run(
                    model, level, sanitize, benchmark_input, adj
                )
            )
            if row[key]["ok"]:
                reference = source_output[level]["raw_sha256"]
                row[key]["result"]["source_raw_sha256"] = reference
                row[key]["result"]["bit_identical_to_source"] = (
                    row[key]["result"]["raw_sha256"] == reference
                )
    row["cost_profile"] = attempt(lambda: cost_profile(model, label))
    row["competition_profile"] = attempt(lambda: competition_profile(model, label))
    row["eligible"] = bool(
        row["competition_profile"]["ok"]
        and row["competition_profile"]["result"]["cost"] < 1195
    )
    row["decision"] = "ELIGIBLE_FOR_DEEP_AUDIT" if row["eligible"] else "REJECT_COST"
    return row


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    source = onnx.load(SOURCE)
    source_hash = sha256(SOURCE)
    adj, _ = source_adj(source)
    source_profile = cost_profile(source, "source")
    benchmark_input = input_array()
    source_output = {
        level: ort_run(source, level, False, benchmark_input, adj)
        for level in ("default", "disabled")
    }
    metadata_modes = [
        "none",
        "dense_value_info",
        "sparse_value_info",
        "dense_graph_input",
        "sparse_graph_input",
    ]
    rows = [
        run_variant(source, source_output, form, metadata)
        for form in ("linear", "coordinates")
        for metadata in metadata_modes
    ]
    controls = [run_control(source, source_output, form) for form in ("linear", "coordinates")]
    eligible = [row["label"] for row in rows + controls if row["eligible"]]
    expected_params = int(
        sum(math.prod(item.dims) for item in source.graph.initializer if item.name != ADJ)
        + np.count_nonzero(adj)
    )
    payload = {
        "task": TASK,
        "source": str(SOURCE.relative_to(REPO)),
        "source_sha256": source_hash,
        "source_profile": source_profile,
        "adj": {
            "shape": list(adj.shape),
            "dense_elements": int(adj.size),
            "nonzero": int(np.count_nonzero(adj)),
            "dense_array_sha256": hashlib.sha256(np.ascontiguousarray(adj).tobytes()).hexdigest(),
        },
        "sparse_initializer_if_supported": {
            "expected_params": expected_params,
            "expected_memory_if_unchanged": source_profile["memory"],
            "expected_cost_if_unchanged": source_profile["memory"] + expected_params,
            "theoretical_score_gain": math.log(
                source_profile["cost"] / (source_profile["memory"] + expected_params)
            ),
        },
        "einsum_schema": schema_evidence(),
        "source_output": source_output,
        "variants": rows,
        "controls": controls,
        "eligible": eligible,
        "deep_audit_run": bool(eligible),
        "known4": "NOT_RUN_NO_ELIGIBLE_LOWER_CANDIDATE" if not eligible else "PENDING",
        "fresh10000": "NOT_RUN_NO_ELIGIBLE_LOWER_CANDIDATE" if not eligible else "PENDING",
        "decision": "REJECT_ALL" if not eligible else "REQUIRES_DEEP_AUDIT",
    }
    out = HERE / "probe_results.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_profile": source_profile,
        "einsum_allows_sparse": payload["einsum_schema"]["allows_sparse_tensor"],
        "variants": len(rows),
        "controls": len(controls),
        "eligible": eligible,
        "result": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
