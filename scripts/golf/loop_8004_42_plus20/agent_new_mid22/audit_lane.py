#!/usr/bin/env python3
"""Audit latest immutable 8005.16 members for the new-mid22 lane.

The script is deliberately read-only outside this directory.  It extracts the
eight exact ZIP members, measures them through the repository scorer, checks
both ORT modes, and records structural/runtime-shape evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8005.16.zip"
TASKS = (123, 316, 212, 301, 55, 86, 163, 206)
CURRENT = HERE / "current"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LOOKUP = {"TFIDFVECTORIZER", "SCATTERELEMENTS", "SCATTERND", "HARDMAX"}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shape_of(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    dims: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


def make_session(
    model: onnx.ModelProto,
    disable: bool,
    *,
    profile_prefix: str | None = None,
) -> tuple[onnx.ModelProto, ort.InferenceSession]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    session = ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    return sanitized, session


def converted_examples(task: int) -> list[dict[str, np.ndarray]]:
    return [
        item
        for split in ("train", "test", "arc-gen")
        for raw in scoring.load_examples(task).get(split, [])
        if (item := scoring.convert_to_numpy(raw)) is not None
    ]


def known_dual(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    examples = converted_examples(task)
    result: dict[str, Any] = {}
    for label, disabled in (("disable_all", True), ("default", False)):
        row: dict[str, Any] = {
            "total": len(examples),
            "right": 0,
            "wrong": 0,
            "runtime_errors": 0,
        }
        try:
            _, session = make_session(model, disabled)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}:{exc}"
            result[label] = row
            continue
        positives: list[float] = []
        unstable = 0
        for item in examples:
            try:
                raw = session.run(None, {"input": item["input"]})[0]
                decoded = (raw > 0).astype(np.float32)
                if np.array_equal(decoded, item["output"]):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                pos = raw[raw > 0]
                if pos.size:
                    positives.append(float(pos.min()))
                unstable += int(bool(np.any((raw > 0) & (raw < 0.25))))
            except Exception:  # noqa: BLE001
                row["runtime_errors"] += 1
        row["min_positive"] = min(positives) if positives else None
        row["unstable_examples"] = unstable
        row["perfect"] = (
            row["right"] == row["total"]
            and row["wrong"] == 0
            and row["runtime_errors"] == 0
        )
        result[label] = row
    return result


def runtime_shapes(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"mid22_shape_{task:03d}_") as tmp:
        sanitized, session = make_session(
            model, True, profile_prefix=str(Path(tmp) / "profile")
        )
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(sanitized), strict_mode=True, data_prop=True
        )
        declared = {
            value.name: shape_of(value)
            for value in list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        }
        item = converted_examples(task)[0]
        output = session.run(None, {"input": item["input"]})[0]
        trace = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        by_name = {node.name: node for node in inferred.graph.node if node.name}
        actual: dict[str, list[int]] = {}
        for event in trace:
            args = event.get("args", {})
            shapes = args.get("output_type_shape")
            if event.get("cat") != "Node" or not shapes:
                continue
            event_name = str(event.get("name", "")).replace("_kernel_time", "")
            node = by_name.get(event_name)
            if node is None:
                continue
            for index, shape_dict in enumerate(shapes):
                if index >= len(node.output) or not node.output[index] or not shape_dict:
                    continue
                dims = next(iter(shape_dict.values()))
                actual.setdefault(node.output[index], [int(value) for value in dims])
        mismatches = {
            name: {"declared": declared.get(name), "runtime": dims}
            for name, dims in actual.items()
            if declared.get(name) != dims
        }
        out_declared = shape_of(inferred.graph.output[0])
        if out_declared != list(output.shape):
            mismatches[inferred.graph.output[0].name] = {
                "declared": out_declared,
                "runtime": list(output.shape),
            }
        return {
            "pass": not mismatches,
            "profiled_outputs": len(actual),
            "mismatches": mismatches,
        }


def score(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"mid22_score_{task:03d}_") as tmp:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, tmp, label="baseline", require_correct=False
        )
    if result is None:
        return {"error": "score_and_verify returned None"}
    return {
        "memory": int(result["memory"]),
        "params": int(result["params"]),
        "cost": int(result["cost"]),
        "score": float(result["score"]),
        "known_correct": bool(result["correct"]),
    }


def structure(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:  # noqa: BLE001
        checker = False
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        strict = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        strict = False
        errors.append(f"shape:{type(exc).__name__}:{exc}")
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    static = all(
        shape_of(value) is not None
        for value in values
        if value.type.HasField("tensor_type")
    )
    try:
        truth = runtime_shapes(model, task)
    except Exception as exc:  # noqa: BLE001
        truth = {"pass": False, "error": f"{type(exc).__name__}:{exc}"}
    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    domains = sorted(
        {node.domain for node in model.graph.node}
        | {item.domain for item in model.opset_import}
    )
    init_arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    used = {name for node in model.graph.node for name in node.input if name}
    graph_outputs = {value.name for value in model.graph.output}
    checks = {
        "checker_full": checker,
        "strict_data_prop": strict,
        "static_positive": static,
        "truthful_runtime_shapes": bool(truth.get("pass")),
        "canonical_io": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and shape_of(model.graph.input[0]) == [1, 10, 30, 30]
            and shape_of(model.graph.output[0]) == [1, 10, 30, 30]
        ),
        "standard_domains": all(domain in ("", "ai.onnx") for domain in domains),
        "no_banned": all(
            node.op_type.upper() not in BANNED
            and "SEQUENCE" not in node.op_type.upper()
            for node in model.graph.node
        ),
        "no_nested_functions_sparse": (
            not model.functions
            and not model.graph.sparse_initializer
            and all(
                attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for node in model.graph.node
                for attr in node.attribute
            )
        ),
        "no_lookup": all(node.op_type.upper() not in LOOKUP for node in model.graph.node),
        "no_shape_cloak": all(node.op_type != "CenterCropPad" for node in model.graph.node),
        "no_giant_einsum": max_einsum <= 16,
        "conv_bias_ub0": not check_conv_bias(model),
        "finite_initializers": all(
            arr.dtype.kind not in "fc" or bool(np.isfinite(arr).all())
            for arr in init_arrays.values()
        ),
    }
    duplicate_groups: list[list[str]] = []
    names = list(init_arrays)
    for i, name in enumerate(names):
        group = [name]
        for other in names[i + 1 :]:
            a, b = init_arrays[name], init_arrays[other]
            if a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b):
                group.append(other)
        if len(group) > 1 and not any(name in prior for prior in duplicate_groups):
            duplicate_groups.append(group)
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_params_static": sum(
            max(1, math.prod(init.dims)) for init in model.graph.initializer
        ),
        "ops": dict(sorted(ops.items())),
        "domains": domains,
        "max_einsum_inputs": max_einsum,
        "unused_initializers": sorted(set(init_arrays) - used - graph_outputs),
        "duplicate_initializers": duplicate_groups,
        "runtime_truth": truth,
        "checks": checks,
        "pass": all(checks.values()),
        "errors": errors,
    }


def main() -> None:
    CURRENT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": sha_bytes(BASE_ZIP.read_bytes()),
        "tasks": {},
    }
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            path = CURRENT / member
            path.write_bytes(data)
            model = onnx.load_from_string(data)
            row = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha_bytes(data),
                "bytes": len(data),
                "score": score(model, task),
                "known_dual": known_dual(model, task),
                "structure": structure(model, task),
            }
            report["tasks"][str(task)] = row
            print(
                f"task{task:03d} sha={row['sha256'][:12]} cost={row['score']} "
                f"known={row['known_dual']['disable_all']['right']}/"
                f"{row['known_dual']['disable_all']['total']} "
                f"struct={row['structure']['pass']}",
                flush=True,
            )
    (HERE / "baseline_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
