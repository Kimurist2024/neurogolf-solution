#!/usr/bin/env python3
"""Fail-closed audit of the eight 8005.16 rebase targets."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "base"
TASKS = (13, 18, 54, 80, 89, 96, 101, 131)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def dims(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def session(model: onnx.ModelProto, optimize: bool) -> ort.InferenceSession:
    clean = scoring.sanitize_model(copy.deepcopy(model))
    if clean is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(clean.SerializeToString(), options)


def known_rows(task: int) -> list[dict[str, np.ndarray]]:
    examples = scoring.load_examples(task)
    rows = []
    for subset in ("train", "test", "arc-gen"):
        for example in examples.get(subset, []):
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append(converted)
    return rows


def verify(sess: ort.InferenceSession, rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    right = errors = near_margin = 0
    min_positive = math.inf
    observed_shapes: Counter[str] = Counter()
    for row in rows:
        try:
            raw = sess.run(["output"], {"input": row["input"]})[0]
        except Exception:
            errors += 1
            continue
        observed_shapes[str(list(raw.shape))] += 1
        expected = row["output"] > 0
        right += int(raw.shape == expected.shape and np.array_equal(raw > 0, expected))
        positive = raw[raw > 0]
        if positive.size:
            min_positive = min(min_positive, float(positive.min()))
        near_margin += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
    return {
        "total": len(rows),
        "right": right,
        "errors": errors,
        "rate": right / len(rows) if rows else None,
        "observed_output_shapes": dict(observed_shapes),
        "near_margin_count": near_margin,
        "min_positive": None if min_positive == math.inf else min_positive,
    }


def runtime_shape_audit(model: onnx.ModelProto, task: int) -> dict[str, Any]:
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=False, data_prop=True)
    typed = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    }
    declared = {
        value.name: dims(value)
        for value in [*model.graph.value_info, *model.graph.output]
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
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(traced.SerializeToString(), options)
    example = known_rows(task)[0]
    values = sess.run(names, {"input": example["input"]})
    actual = {name: list(np.asarray(value).shape) for name, value in zip(names, values)}
    mismatches = [
        {"tensor": name, "declared": declared[name], "runtime": actual[name]}
        for name in declared
        if name in actual and declared[name] != actual[name]
    ]
    return {
        "truthful": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    report: dict[str, Any] = {}
    for task in TASKS:
        path = BASE / f"task{task:03d}.onnx"
        raw = path.read_bytes()
        model = onnx.load_from_string(raw)
        entry: dict[str, Any] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "file_size": len(raw),
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "op_histogram": dict(Counter(node.op_type for node in model.graph.node)),
            "max_einsum_inputs": max(
                (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
                default=0,
            ),
            "center_crop_pad_nodes": sum(node.op_type == "CenterCropPad" for node in model.graph.node),
            "lookup_ops": [node.op_type for node in model.graph.node if node.op_type in {"TfIdfVectorizer", "Hardmax"}],
            "standard_domains": all(item.domain in {"", "ai.onnx"} for item in model.opset_import)
            and all(node.domain in {"", "ai.onnx"} for node in model.graph.node),
            "no_banned_or_sequence": all(
                node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper()
                for node in model.graph.node
            ),
            "no_nested_graphs": all(
                attr.type not in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
                for node in model.graph.node
                for attr in node.attribute
            )
            and not model.functions,
            "conv_bias_findings": check_conv_bias(model),
        }
        try:
            onnx.checker.check_model(model, full_check=True)
            entry["checker_full"] = True
        except Exception as exc:
            entry["checker_full"] = False
            entry["checker_error"] = f"{type(exc).__name__}: {exc}"
        try:
            inferred = shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=True
            )
            entry["strict_data_prop"] = True
            values = [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
            entry["static_positive"] = all(dims(value) is not None for value in values)
        except Exception as exc:
            entry["strict_data_prop"] = False
            entry["static_positive"] = False
            entry["strict_error"] = f"{type(exc).__name__}: {exc}"
        memory, params, cost = cost_of(str(path))
        entry["measured"] = {"memory": memory, "params": params, "cost": cost}
        rows = known_rows(task)
        entry["known"] = {}
        for optimize, label in ((False, "disable_all"), (True, "default")):
            try:
                entry["known"][label] = verify(session(model, optimize), rows)
            except Exception as exc:
                entry["known"][label] = {"session_error": f"{type(exc).__name__}: {exc}"}
        try:
            entry["runtime_shapes"] = runtime_shape_audit(model, task)
        except Exception as exc:
            entry["runtime_shapes"] = {
                "truthful": False,
                "trace_error": f"{type(exc).__name__}: {exc}",
            }
        report[f"task{task:03d}"] = entry
        (HERE / "baseline_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"task{task:03d} cost={cost} known={entry['known']} "
            f"truthful={entry['runtime_shapes']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
