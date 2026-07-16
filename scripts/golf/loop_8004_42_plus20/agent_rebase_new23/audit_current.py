#!/usr/bin/env python3
"""Fail-closed audit of the eight 8005.16 changed members assigned to this lane.

This script is deliberately non-promoting.  It extracts only the assigned
members into this lane, remeasures actual cost, checks static structure and
Conv-family bias lengths, and runs the complete stored corpus in both ORT
modes.  The immutable baseline ZIP and shared score/submission files are never
written.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
TARGETS = (133, 145, 182, 187, 201, 204, 216, 233)
BASELINE_DIR = HERE / "baseline"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append("?")
    return result


def known_rows(task: int) -> list[dict[str, np.ndarray]]:
    loaded = scoring.load_examples(task)
    rows: list[dict[str, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for example in loaded.get(split, []):
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append(converted)
    return rows


def make_session(model: onnx.ModelProto, mode: str) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "disable_all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def run_known(model: onnx.ModelProto, task: int, mode: str) -> dict[str, object]:
    rows = known_rows(task)
    result: dict[str, object] = {
        "total": len(rows),
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "runtime_output_shapes": [],
    }
    try:
        session = make_session(model, mode)
    except Exception as exc:  # noqa: BLE001
        result["session_error"] = f"{type(exc).__name__}: {exc}"
        return result
    output_shapes: set[tuple[int, ...]] = set()
    first_error: str | None = None
    for row in rows:
        try:
            raw = session.run(["output"], {"input": row["input"]})[0]
            output_shapes.add(tuple(int(value) for value in raw.shape))
            correct = np.array_equal(raw > 0, row["output"] > 0)
            result["right"] = int(result["right"]) + int(correct)
            result["wrong"] = int(result["wrong"]) + int(not correct)
        except Exception as exc:  # noqa: BLE001
            result["runtime_errors"] = int(result["runtime_errors"]) + 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
    result["runtime_output_shapes"] = [list(item) for item in sorted(output_shapes)]
    if first_error is not None:
        result["first_runtime_error"] = first_error
    result["perfect"] = bool(
        result["right"] == result["total"] and result["runtime_errors"] == 0
    )
    return result


def structure(model: onnx.ModelProto) -> dict[str, object]:
    report: dict[str, object] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        report["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        report["checker_full"] = False
        report["checker_error"] = f"{type(exc).__name__}: {exc}"
    try:
        inferred = shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        report["strict_data_prop"] = True
        report["inferred_output_shape"] = dims(inferred.graph.output[0])
    except Exception as exc:  # noqa: BLE001
        report["strict_data_prop"] = False
        report["strict_error"] = f"{type(exc).__name__}: {exc}"
    report["declared_input_shape"] = dims(model.graph.input[0])
    report["declared_output_shape"] = dims(model.graph.output[0])
    report["standard_domains"] = all(
        item.domain in ("", "ai.onnx") for item in model.opset_import
    ) and all(node.domain in ("", "ai.onnx") for node in model.graph.node)
    report["functions"] = len(model.functions)
    report["sparse_initializers"] = len(model.graph.sparse_initializer)
    report["nested_graph_attributes"] = sum(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    report["banned_ops"] = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
        }
    )
    report["op_histogram"] = dict(sorted(Counter(node.op_type for node in model.graph.node).items()))
    report["lookup_red_flags"] = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in {
                "TfIdfVectorizer",
                "Hardmax",
                "GatherND",
                "ScatterElements",
                "ScatterND",
            }
        }
    )
    report["conv_bias_ub"] = [list(item) for item in check_conv_bias(model)]
    init = {item.name for item in model.graph.initializer}
    report["dynamic_conv_biases"] = [
        {
            "node": node.name,
            "op": node.op_type,
            "bias": node.input[8 if node.op_type == "QLinearConv" else 2],
        }
        for node in model.graph.node
        if node.op_type in {"Conv", "ConvTranspose", "QLinearConv"}
        and len(node.input) > (8 if node.op_type == "QLinearConv" else 2)
        and node.input[8 if node.op_type == "QLinearConv" else 2]
        and node.input[8 if node.op_type == "QLinearConv" else 2] not in init
    ]
    return report


def main() -> int:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    archive_sha = sha(BASE.read_bytes())
    rows = []
    with zipfile.ZipFile(BASE) as archive:
        for task in TARGETS:
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            path = BASELINE_DIR / member
            path.write_bytes(data)
            model = onnx.load_model_from_string(data)
            memory, params, cost = cost_of(str(path))
            record = {
                "task": task,
                "member": member,
                "sha256": sha(data),
                "bytes": len(data),
                "nodes": len(model.graph.node),
                "initializers": len(model.graph.initializer),
                "cost": {"memory": memory, "params": params, "cost": cost},
                "structure": structure(model),
                "known": {
                    mode: run_known(model, task, mode)
                    for mode in ("disable_all", "default")
                },
            }
            rows.append(record)
            print(
                f"task{task:03d} cost={cost} sha={record['sha256'][:12]} "
                f"known={record['known']['disable_all'].get('right')}/"
                f"{record['known']['disable_all'].get('total')} "
                f"default={record['known']['default'].get('right')}/"
                f"{record['known']['default'].get('total')}",
                flush=True,
            )
    payload = {
        "baseline": str(BASE.relative_to(ROOT)),
        "baseline_sha256": archive_sha,
        "targets": list(TARGETS),
        "records": rows,
    }
    (HERE / "baseline_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
