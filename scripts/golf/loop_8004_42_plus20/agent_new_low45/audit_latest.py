#!/usr/bin/env python3
"""Read-only structural audit for the low45 eight-target expansion."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
PREVIOUS = ROOT / "submission_base_8004.50.zip"
TARGETS = (24, 113, 385, 389, 296, 399, 359, 110)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_private_exact15.audit_exact import (  # noqa: E402
    trace_shapes,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    dims: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


def static_cost(inferred: onnx.ModelProto) -> dict[str, int]:
    infos = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    excluded = {value.name for value in inferred.graph.input}
    excluded.update(value.name for value in inferred.graph.output)
    excluded.update(item.name for item in inferred.graph.initializer)
    memory = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in excluded or name in seen:
                continue
            seen.add(name)
            value = infos.get(name)
            dims = tensor_shape(value) if value is not None else None
            if dims is None:
                raise RuntimeError(f"non-static node output: {name}")
            dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            memory += math.prod(dims) * np.dtype(dtype).itemsize
    params = sum(math.prod(item.dims) if item.dims else 1 for item in inferred.graph.initializer)
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def structural(model: onnx.ModelProto, task: int) -> dict[str, object]:
    errors: list[str] = []
    checker = False
    strict = False
    inferred = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        strict = True
    except Exception as exc:
        errors.append(f"strict_data_prop:{type(exc).__name__}:{exc}")
    inspected = inferred if inferred is not None else model
    values = list(inspected.graph.input) + list(inspected.graph.value_info) + list(inspected.graph.output)
    static_positive = all(tensor_shape(value) is not None for value in values)
    nonstandard_domains = sorted(
        {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
        | {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
    )
    ops = collections.Counter(node.op_type for node in model.graph.node)
    giant_einsum = [
        {"node_index": index, "inputs": len(node.input)}
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Einsum" and len(node.input) > 16
    ]
    lookup_nodes = [
        node.op_type
        for node in model.graph.node
        if node.op_type in {"TfIdfVectorizer", "Hardmax", "GatherND", "ScatterND", "ScatterElements"}
    ]
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        }
    )
    declared_cost = None
    if inferred is not None:
        try:
            declared_cost = static_cost(inferred)
        except Exception as exc:
            errors.append(f"static_cost:{type(exc).__name__}:{exc}")
    try:
        shapes = trace_shapes(model, task)
    except Exception as exc:
        shapes = {"error": f"{type(exc).__name__}: {exc}", "shape_cloak": True}
    return {
        "checker_full": checker,
        "strict_shape_data_prop": strict,
        "static_positive": static_positive,
        "standard_domains": not nonstandard_domains,
        "nonstandard_domains": nonstandard_domains,
        "banned_ops": banned,
        "conv_bias_findings": check_conv_bias(model),
        "op_histogram": dict(sorted(ops.items())),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        "giant_einsum": giant_einsum,
        "lookup_or_scatter_nodes": lookup_nodes,
        "declared_cost": declared_cost,
        "runtime_shape_trace": shapes,
        "errors": errors,
    }


def main() -> None:
    rows = []
    with zipfile.ZipFile(BASE) as current, zipfile.ZipFile(PREVIOUS) as previous:
        for task in TARGETS:
            member = f"task{task:03d}.onnx"
            data = current.read(member)
            old = previous.read(member)
            model = onnx.load_model_from_string(data)
            (HERE / "baselines" / member).write_bytes(data)
            with tempfile.TemporaryDirectory(prefix=f"low45_{task}_", dir="/tmp") as temp:
                path = Path(temp) / member
                path.write_bytes(data)
                memory, params, cost = cost_of(str(path))
            row = {
                "task": task,
                "member": member,
                "sha256": sha(data),
                "file_bytes": len(data),
                "unchanged_from_8004_50": data == old,
                "previous_member_sha256": sha(old),
                "actual_cost": {"memory": memory, "params": params, "cost": cost},
                "structure": structural(model, task),
            }
            rows.append(row)
            print(
                f"task{task:03d}: cost={cost} ops={row['structure']['op_histogram']} "
                f"unchanged={data == old}",
                flush=True,
            )
    output = {
        "baseline": {
            "path": BASE.name,
            "sha256": sha(BASE.read_bytes()),
            "previous_path": PREVIOUS.name,
            "previous_sha256": sha(PREVIOUS.read_bytes()),
        },
        "targets": rows,
    }
    (HERE / "baseline_audit.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
