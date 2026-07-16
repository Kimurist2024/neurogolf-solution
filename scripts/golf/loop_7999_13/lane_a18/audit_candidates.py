#!/usr/bin/env python3
"""Strict structural, known-set, cost, cloak, UB, lookup, and diff audit."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


BASE_COST = {63: 26, 73: 16, 139: 52, 202: 48}
CANDIDATES = (
    (63, "r01"),
    (73, "r01"),
    (73, "r02"),
    (139, "r04"),
    (202, "r02"),
    (202, "r03"),
)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    if not value.type.HasField("tensor_type"):
        return ["non_tensor"]
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("?")
    return result


def equation(node: onnx.NodeProto) -> str | None:
    if node.op_type != "Einsum":
        return None
    for attribute in node.attribute:
        if attribute.name == "equation":
            value = helper.get_attribute_value(attribute)
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return None


def initializer_rows(model: onnx.ModelProto) -> list[dict[str, object]]:
    rows = []
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        raw = array.tobytes()
        rows.append(
            {
                "name": initializer.name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "elements": int(array.size),
                "content_sha256": sha(raw),
                "finite": bool(np.isfinite(array).all())
                if array.dtype.kind in "fc"
                else True,
            }
        )
    return rows


def make_session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )


def known(task: int, session: ort.InferenceSession) -> dict[str, object]:
    rows: dict[str, dict[str, int]] = {}
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for subset in ("train", "test", "arc-gen"):
        right = wrong = errors = 0
        for example in scoring.load_examples(task)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                actual = session.run(
                    [output_name], {input_name: benchmark["input"]}
                )[0]
                if np.array_equal(actual > 0, benchmark["output"] > 0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001
                errors += 1
        rows[subset] = {"right": right, "wrong": wrong, "errors": errors}
    rows["total"] = {
        key: sum(row[key] for name, row in rows.items() if name != "total")
        for key in ("right", "wrong", "errors")
    }
    return rows


def canonical_executable_bytes(model: onnx.ModelProto) -> bytes:
    clone = copy.deepcopy(model)
    clone.producer_name = ""
    clone.producer_version = ""
    clone.domain = ""
    clone.model_version = 0
    clone.doc_string = ""
    del clone.metadata_props[:]
    clone.graph.name = ""
    clone.graph.doc_string = ""
    for node in clone.graph.node:
        node.name = ""
        node.doc_string = ""
    # Normalize equivalent TensorProto storage (typed fields versus raw_data).
    for index, initializer in enumerate(list(clone.graph.initializer)):
        normalized = numpy_helper.from_array(
            numpy_helper.to_array(initializer), initializer.name
        )
        clone.graph.initializer[index].CopyFrom(normalized)
    return clone.SerializeToString(deterministic=True)


def inspect_model(task: int, label: str, path: Path) -> dict[str, object]:
    model = onnx.load(path)
    data = path.read_bytes()
    ops = Counter(node.op_type for node in model.graph.node)
    row: dict[str, object] = {
        "task": task,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(data),
        "file_bytes": len(data),
        "node_count": len(model.graph.node),
        "ops": dict(ops),
        "node_inputs": [len(node.input) for node in model.graph.node],
        "equations": [value for node in model.graph.node if (value := equation(node))],
        "initializer_rows": initializer_rows(model),
        "params": scoring.calculate_params(model),
        "value_info_count": len(model.graph.value_info),
        "input_shapes": [dims(value) for value in model.graph.input],
        "declared_output_shapes": [dims(value) for value in model.graph.output],
        "nonstandard_opsets": [
            item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")
        ],
        "nonstandard_nodes": [
            node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")
        ],
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": [
            item.name
            for item in model.graph.initializer
            if item.data_location == onnx.TensorProto.EXTERNAL or item.external_data
        ],
        "nested_graph_ops": [
            node.op_type
            for node in model.graph.node
            for attr in node.attribute
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        ],
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        ],
        "conv_bias_issues": check_conv_bias(model),
        "lookup_red_flags": {
            "tfidf": int(ops.get("TfIdfVectorizer", 0)),
            "hardmax": int(ops.get("Hardmax", 0)),
            "onehot": int(ops.get("OneHot", 0)),
            "large_initializer_elements": sum(
                int(numpy_helper.to_array(item).size) >= 1000
                for item in model.graph.initializer
            ),
        },
        "giant_einsum_inputs": max(
            (
                len(node.input)
                for node in model.graph.node
                if node.op_type == "Einsum"
            ),
            default=0,
        ),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, full_check_error=f"{type(exc).__name__}: {exc}")
    inferred = None
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        row["strict_shape_data_prop"] = True
        row["inferred_output_shapes"] = [dims(value) for value in inferred.graph.output]
        row["nonstatic_inferred_tensors"] = [
            value.name
            for value in list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
            if "?" in dims(value) or "non_tensor" in dims(value)
        ]
    except Exception as exc:  # noqa: BLE001
        row.update(
            strict_shape_data_prop=False,
            strict_shape_error=f"{type(exc).__name__}: {exc}",
        )

    runtime_shapes = []
    runtime_error = None
    try:
        session = make_session(model, True)
        first = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
        if first is None:
            raise RuntimeError("first known example unavailable")
        outputs = session.run(None, {session.get_inputs()[0].name: first["input"]})
        runtime_shapes = [list(np.asarray(output).shape) for output in outputs]
    except Exception as exc:  # noqa: BLE001
        runtime_error = f"{type(exc).__name__}: {exc}"
    row["runtime_output_shapes"] = runtime_shapes
    row["runtime_shape_error"] = runtime_error
    row["shape_cloak_free"] = bool(
        not model.graph.value_info
        and runtime_error is None
        and runtime_shapes == row["declared_output_shapes"]
        and (
            inferred is not None
            and row.get("inferred_output_shapes") == row["declared_output_shapes"]
        )
    )

    ub_issues = []
    if row["conv_bias_issues"]:
        ub_issues.append("conv_bias_length_mismatch")
    if row["external_initializers"]:
        ub_issues.append("external_initializer")
    if any(not item["finite"] for item in row["initializer_rows"]):
        ub_issues.append("nonfinite_initializer")
    if any(
        dimension <= 0
        for item in model.graph.initializer
        for dimension in item.dims
    ):
        ub_issues.append("nonpositive_initializer_dimension")
    row["ub_issues"] = ub_issues

    with tempfile.TemporaryDirectory(prefix=f"a18_{label}_", dir="/tmp") as workdir:
        try:
            row["official_like_score"] = scoring.score_and_verify(
                copy.deepcopy(model), task, workdir, label, require_correct=False
            )
        except Exception as exc:  # noqa: BLE001
            row["official_like_score_error"] = f"{type(exc).__name__}: {exc}"
    for disable_all, key in ((True, "known_disable_all"), (False, "known_default")):
        try:
            row[key] = known(task, make_session(model, disable_all))
        except Exception as exc:  # noqa: BLE001
            row[key] = {"session_error": f"{type(exc).__name__}: {exc}"}
    return row


def diff(base: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, object]:
    base_ops = [(node.op_type, node.domain) for node in base.graph.node]
    candidate_ops = [(node.op_type, node.domain) for node in candidate.graph.node]
    return {
        "node_sequence_identical": base_ops == candidate_ops,
        "node_count_delta": len(candidate.graph.node) - len(base.graph.node),
        "parameter_delta": scoring.calculate_params(candidate)
        - scoring.calculate_params(base),
        "value_info_delta": len(candidate.graph.value_info) - len(base.graph.value_info),
        "input_contract_identical": base.graph.input == candidate.graph.input,
        "output_contract_identical": base.graph.output == candidate.graph.output,
        "base_equations": [
            value for node in base.graph.node if (value := equation(node))
        ],
        "candidate_equations": [
            value for node in candidate.graph.node if (value := equation(node))
        ],
        "base_initializers": initializer_rows(base),
        "candidate_initializers": initializer_rows(candidate),
    }


def pre_fresh_pass(row: dict[str, object]) -> tuple[bool, list[str]]:
    reasons = []
    score = row.get("official_like_score")
    if not score or not score.get("correct"):
        reasons.append("known_or_profile_fail")
    elif score["cost"] >= BASE_COST[int(row["task"])]:
        reasons.append("not_strictly_cheaper")
    for field in ("full_check", "strict_shape_data_prop", "shape_cloak_free"):
        if not row.get(field):
            reasons.append(field)
    for field in (
        "nonstandard_opsets",
        "nonstandard_nodes",
        "functions",
        "sparse_initializers",
        "external_initializers",
        "nested_graph_ops",
        "banned_ops",
        "conv_bias_issues",
        "ub_issues",
        "nonstatic_inferred_tensors",
    ):
        if row.get(field):
            reasons.append(field)
    if any(row["lookup_red_flags"].values()):
        reasons.append("lookup_red_flags")
    for field in ("known_disable_all", "known_default"):
        total = row.get(field, {}).get("total", {})
        if total.get("wrong") != 0 or total.get("errors") != 0 or not total.get("right"):
            reasons.append(field)
    return not reasons, reasons


def main() -> None:
    ort.set_default_logger_severity(4)
    baselines: dict[str, object] = {}
    candidates: dict[str, object] = {}
    diffs: dict[str, object] = {}
    base_models = {
        task: onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
        for task in BASE_COST
    }
    for task in BASE_COST:
        label = f"task{task:03d}_base"
        row = inspect_model(task, label, HERE / "baseline" / f"task{task:03d}.onnx")
        baselines[label] = row
        print(label, row.get("official_like_score"), flush=True)
    for task, variant in CANDIDATES:
        label = f"task{task:03d}_{variant}"
        path = HERE / "candidates" / f"{label}.onnx"
        row = inspect_model(task, label, path)
        passed, reasons = pre_fresh_pass(row)
        row["pre_fresh_pass"] = passed
        row["pre_fresh_reasons"] = reasons
        candidates[label] = row
        candidate_model = onnx.load(path)
        diffs[label] = diff(base_models[task], candidate_model)
        print(
            label,
            row.get("official_like_score"),
            "pass=" + str(passed),
            reasons,
            flush=True,
        )
    r1 = onnx.load(HERE / "candidates/task073_r01.onnx")
    r2 = onnx.load(HERE / "candidates/task073_r02.onnx")
    variant_comparison = {
        "task073_r01_r02_executable_identical_after_metadata_clear": (
            canonical_executable_bytes(r1) == canonical_executable_bytes(r2)
        ),
        "task073_nonsemantic_differences": [
            "graph.name",
            "initializer protobuf storage encoding (raw_data versus float_data)",
        ],
    }
    output = {
        "baseline_costs": BASE_COST,
        "baselines": baselines,
        "candidates": candidates,
        "diffs": diffs,
        "variant_comparison": variant_comparison,
    }
    (HERE / "candidate_audit.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
