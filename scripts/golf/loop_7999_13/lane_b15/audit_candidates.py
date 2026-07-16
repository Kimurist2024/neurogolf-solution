#!/usr/bin/env python3
"""Audit all archived below-baseline B15 models without promoting anything."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
BASE_COST = {23: 1622, 36: 325}
CANDIDATES = {
    23: [
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task023_r01_static1497.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task023_r02_static1520.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task023_r03_static1541.onnx",
    ],
    36: [
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task036_r01_static212.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task036_r02_static214.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task036_r03_static230.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task036_r04_static231.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task036_r05_static232.onnx",
        ROOT / "others/7907/task036_rebuild_attempt.onnx",
        HERE / "candidate_task036_truthful_gather.onnx",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dimension.dim_value)
        if dimension.HasField("dim_value")
        else dimension.dim_param or "?"
        for dimension in value.type.tensor_type.shape.dim
    ]


def static_shape_ok(value: onnx.ValueInfoProto) -> bool:
    return all(
        dimension.HasField("dim_value") and dimension.dim_value > 0
        for dimension in value.type.tensor_type.shape.dim
    )


def structural(model: onnx.ModelProto) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    error = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checks["checker_full"] = True
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        checks["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001 - rejection evidence
        checks["checker_full"] = False
        checks["strict_data_prop"] = False
        inferred = model
        error = f"{type(exc).__name__}: {exc}"
    checks["canonical_io"] = (
        len(model.graph.input) == 1
        and len(model.graph.output) == 1
        and model.graph.input[0].name == "input"
        and model.graph.output[0].name == "output"
    )
    checks["standard_domains"] = all(
        item.domain in ("", "ai.onnx") for item in model.opset_import
    ) and all(node.domain in ("", "ai.onnx") for node in model.graph.node)
    checks["no_functions_sparse_nested"] = (
        not model.functions
        and not model.graph.sparse_initializer
        and all(
            attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node
            for attribute in node.attribute
        )
    )
    checks["no_banned_ops"] = all(
        node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper()
        for node in model.graph.node
    )
    checks["no_giant_einsum"] = all(
        node.op_type != "Einsum" or len(node.input) <= 16 for node in model.graph.node
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    checks["static_positive_shapes"] = all(static_shape_ok(value) for value in values)
    checks["no_external_initializers"] = all(
        item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data
        for item in model.graph.initializer
    )
    checks["finite_initializers"] = all(
        array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
        for array in (numpy_helper.to_array(item) for item in model.graph.initializer)
    )
    checks["conv_bias_safe"] = not check_conv_bias(model)
    return {"checks": checks, "pass_before_runtime_shape": all(checks.values()), "error": error}


def trace_runtime_shapes(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    }
    declared = {
        value.name: declared_shape(value)
        for value in list(model.graph.output) + list(model.graph.value_info)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    sess = ort.InferenceSession(traced.SerializeToString(), options)
    example = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    assert example is not None
    outputs = sess.run(names, {"input": example["input"]})
    arrays = {name: np.asarray(value) for name, value in zip(names, outputs)}
    actual = {name: list(array.shape) for name, array in arrays.items()}
    mismatches = [
        {"tensor": name, "declared": shape, "actual": actual[name]}
        for name, shape in declared.items()
        if name in actual and shape != actual[name]
    ]
    graph_outputs = {value.name for value in model.graph.output}
    truthful_bytes = sum(array.nbytes for name, array in arrays.items() if name not in graph_outputs)
    return {
        "declared_runtime_mismatches": mismatches,
        "shape_cloak": bool(mismatches),
        "truthful_one_example_intermediate_bytes": int(truthful_bytes),
    }


def make_session(model: onnx.ModelProto, mode: str) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known_dual(model: onnx.ModelProto, task: int) -> list[dict[str, Any]]:
    examples = scoring.load_examples(task)
    rows = []
    for mode in ("disabled", "default"):
        right = wrong = errors = 0
        first_failure = None
        try:
            sess = make_session(model, mode)
        except Exception as exc:  # noqa: BLE001
            rows.append({"mode": mode, "right": 0, "wrong": 0, "errors": 1, "session_error": repr(exc)})
            continue
        for subset in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[subset]):
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    continue
                try:
                    raw = sess.run(["output"], {"input": benchmark["input"]})[0]
                    if np.array_equal(raw > 0.0, benchmark["output"].astype(bool)):
                        right += 1
                    else:
                        wrong += 1
                        first_failure = first_failure or {"subset": subset, "index": index}
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    first_failure = first_failure or {"subset": subset, "index": index, "error": repr(exc)}
        rows.append({"mode": mode, "right": right, "wrong": wrong, "errors": errors, "first_failure": first_failure})
    return rows


def actual_score(model: onnx.ModelProto, task: int, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"b15_{task:03d}_{label}_", dir="/tmp") as workdir:
        return scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, label=label, require_correct=False
        )


def semantic_rejection(task: int, model: onnx.ModelProto, digest: str) -> list[str]:
    reasons: list[str] = []
    ops = {node.op_type for node in model.graph.node}
    if task == 23:
        reasons.append("generator_non_injective")
        if digest == "61313447a8f811f65257ae079330a956e63e8722daf4197f62316649e31798a7":
            reasons.append("parent_excluded_cost1497_family_fresh13_of_5000")
        if digest == "d4eecc87eaa0abf3ff9425153cda8d645db4b224f4c0b7590c62cc443a685a72":
            reasons.append("known_PRIVATE0_cost1520")
        if "GatherND" in ops or ("TopK" in ops and "ScatterElements" in ops):
            reasons.append("lookup_or_rank_scatter_memorization")
        if "Gather" in ops and "BitShift" in ops and "BitwiseXor" in ops:
            reasons.append("fixed_hash_correction_memorization")
    if task == 36:
        reasons.append("approximate_compactness_selector_not_exact_connected_component_rule")
    return reasons


def main() -> int:
    ort.set_default_logger_severity(4)
    rows: list[dict[str, Any]] = []
    for task, paths in CANDIDATES.items():
        seen: set[str] = set()
        for path in paths:
            if not path.exists():
                continue
            digest = sha256(path)
            if digest in seen:
                continue
            seen.add(digest)
            model = onnx.load(path)
            structure = structural(model)
            try:
                runtime_shape = trace_runtime_shapes(model, task)
            except Exception as exc:  # noqa: BLE001 - trace failure rejects shape safety
                runtime_shape = {"shape_cloak": None, "trace_error": f"{type(exc).__name__}: {exc}"}
            score = actual_score(model, task, digest[:8])
            dual = known_dual(model, task)
            semantic = semantic_rejection(task, model, digest)
            known_perfect = all(item["wrong"] == 0 and item["errors"] == 0 for item in dual)
            eligible = (
                structure["pass_before_runtime_shape"]
                and runtime_shape.get("shape_cloak") is False
                and score is not None
                and score["cost"] < BASE_COST[task]
                and known_perfect
                and not semantic
            )
            rows.append(
                {
                    "task": task,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest,
                    "nodes": len(model.graph.node),
                    "initializers": len(model.graph.initializer),
                    "structure": structure,
                    "runtime_shape": runtime_shape,
                    "actual_score": score,
                    "known_dual": dual,
                    "semantic_rejections": semantic,
                    "eligible_for_fresh5000": eligible,
                }
            )
            print(task, path.name, score, "known", [(x["right"], x["wrong"], x["errors"]) for x in dual], "eligible", eligible, flush=True)
    report = {
        "base_cost": BASE_COST,
        "rows": rows,
        "eligible_for_fresh5000": [row for row in rows if row["eligible_for_fresh5000"]],
        "winners_pre_fresh": [],
    }
    (HERE / "candidate_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
