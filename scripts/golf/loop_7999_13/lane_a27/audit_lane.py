#!/usr/bin/env python3
"""Reproducible structural/cost/history audit for A27 tasks 354 and 368."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


BASE_SHA = {
    354: "c86ec60a3cf1241903cd6ebdf11f210f24f9e57216927c5654985d8f2d28efe4",
    368: "0d950f5053aa62e7a3208be01514ad061b85580875e0e93aa7ee941cbacaa811",
}
EXPECTED_COST = {354: 537, 368: 521}
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    dims: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(dim.dim_param or "?")
    return dims


def static_floor(model: onnx.ModelProto) -> tuple[int | None, int]:
    try:
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        assert sanitized is not None
        inferred = onnx.shape_inference.infer_shapes(
            sanitized, strict_mode=True, data_prop=True
        )
    except Exception:  # noqa: BLE001
        return None, scoring.calculate_params(model)
    total = 0
    for value in inferred.graph.value_info:
        dims = shape(value)
        if any(not isinstance(dim, int) or dim <= 0 for dim in dims):
            return None, scoring.calculate_params(model)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
        total += int(math.prod(dims) * np.dtype(dtype).itemsize)
    return total, scoring.calculate_params(model)


def reachability(model: onnx.ModelProto) -> dict[str, object]:
    producer = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    live_names = {value.name for value in model.graph.output}
    live_node_ids: set[int] = set()
    stack = list(live_names)
    while stack:
        name = stack.pop()
        node = producer.get(name)
        if node is None or id(node) in live_node_ids:
            continue
        live_node_ids.add(id(node))
        stack.extend(item for item in node.input if item)
    return {
        "nodes": len(model.graph.node),
        "live_nodes": len(live_node_ids),
        "dead_nodes": len(model.graph.node) - len(live_node_ids),
    }


def duplicate_initializers(model: onnx.ModelProto) -> list[list[str]]:
    groups: dict[tuple[int, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for item in model.graph.initializer:
        array = onnx.numpy_helper.to_array(item)
        groups[(item.data_type, tuple(array.shape), array.tobytes())].append(item.name)
    return [names for names in groups.values() if len(names) > 1]


def structural(path: Path, task: int) -> dict[str, object]:
    model = onnx.load(path)
    memory, params = static_floor(model)
    record: dict[str, object] = {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "nodes": len(model.graph.node),
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node)),
        "params": params,
        "declared_static_memory": memory,
        "declared_static_cost": None if memory is None else memory + params,
        "output_shapes": [shape(value) for value in model.graph.output],
        "reachability": reachability(model),
        "duplicate_initializers": duplicate_initializers(model),
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        ],
        "nested_graphs": sum(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for node in model.graph.node
            for attr in node.attribute
        ),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        record["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        record["full_check"] = False
        record["full_check_error"] = repr(exc)
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        record["strict_shape_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        record["strict_shape_data_prop"] = False
        record["strict_shape_error"] = repr(exc)
    witnesses: list[dict[str, object]] = []
    for value in model.graph.value_info:
        if value.name in {"gn_f", "data_clear_f", "gn"} and shape(value) == [1, 1, 1, 1]:
            witnesses.append({
                "tensor": value.name,
                "declared": [1, 1, 1, 1],
                "guaranteed_runtime": [1, 10, 30, 30],
                "reason": "GroupNormalization preserves the fixed input shape",
            })
    if task == 354 and shape(model.graph.output[0]) != [1, 10, 30, 30]:
        witnesses.append({
            "tensor": "output",
            "declared": shape(model.graph.output[0]),
            "guaranteed_runtime": [1, 10, 30, 30],
            "reason": "known examples and ScatterElements path return a full padded grid",
        })
    record["shape_cloak_witnesses"] = witnesses
    record["shape_cloak_free"] = not witnesses
    return record


def history(task: int) -> dict[str, object]:
    summary = json.loads(
        (HERE.parent / "lane_c9" / "history_summary.json").read_text(encoding="utf-8")
    )[str(task)]
    return {
        "rows": summary["rows"],
        "stage_counts": summary["stage_counts"],
        "lowest_actual": summary["lowest_actual"],
        "lowest_static_floor": summary["lowest_static_floor"],
    }


def main() -> None:
    bases = {}
    for task in (354, 368):
        path = HERE / "base" / f"task{task}.onnx"
        bases[str(task)] = structural(path, task)
        assert bases[str(task)]["sha256"] == BASE_SHA[task]
        assert bases[str(task)]["declared_static_cost"] == EXPECTED_COST[task]
    probes = {
        "task354_no_identity": structural(
            HERE / "candidates" / "task354_no_identity.onnx", 354
        ),
        "task354_archive560": structural(
            HERE / "candidates" / "task354_archive531.onnx", 354
        ),
    }
    payload = {
        "exact_wave12_baselines": bases,
        "algebraic_probes": probes,
        "history": {str(task): history(task) for task in (354, 368)},
        "generator_rules": {
            "354": "replace each gray rectangle by the color of its uniquely aligned top-row light",
            "368": "copy the unique colored prototype sprite into every gray sprite footprint",
        },
        "decision": {
            "accepted": [],
            "projected_gain": 0.0,
            "reason": "no candidate is both strictly cheaper and free of shape cloaking",
        },
    }
    output = HERE / "evidence" / "audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
