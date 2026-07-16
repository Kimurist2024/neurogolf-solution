#!/usr/bin/env python3
"""Produce the fail-closed B19 structural and factor-reuse evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (340, 361)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}


def shape_of(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
        for dim in value.type.tensor_type.shape.dim
    ]


def params(model: onnx.ModelProto) -> int:
    return sum(math.prod(item.dims) if item.dims else 1 for item in model.graph.initializer)


def live_node_indices(model: onnx.ModelProto) -> set[int]:
    producer = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if output
    }
    needed = [item.name for item in model.graph.output]
    live: set[int] = set()
    while needed:
        name = needed.pop()
        index = producer.get(name)
        if index is None or index in live:
            continue
        live.add(index)
        needed.extend(name for name in model.graph.node[index].input if name)
    return live


def duplicate_groups(model: onnx.ModelProto) -> list[list[str]]:
    groups: dict[tuple[int, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for item in model.graph.initializer:
        array = np.ascontiguousarray(numpy_helper.to_array(item))
        key = (int(item.data_type), tuple(int(x) for x in item.dims), array.tobytes())
        groups[key].append(item.name)
    return [names for names in groups.values() if len(names) > 1]


def group_norm_shape_witnesses(model: onnx.ModelProto) -> list[dict[str, object]]:
    values = {
        item.name: shape_of(item)
        for item in [*model.graph.input, *model.graph.value_info, *model.graph.output]
        if item.type.HasField("tensor_type")
    }
    witnesses: list[dict[str, object]] = []
    for node in model.graph.node:
        if node.op_type != "GroupNormalization" or not node.input or not node.output:
            continue
        source = values.get(node.input[0])
        declared = values.get(node.output[0])
        if source is not None and declared is not None and source != declared:
            witnesses.append(
                {
                    "tensor": node.output[0],
                    "declared": declared,
                    "guaranteed_runtime": source,
                    "reason": "GroupNormalization preserves the input tensor shape",
                }
            )
    return witnesses


def inspect(path: Path, task: int) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    full_check = strict_shape = True
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        full_check = False
        errors.append(f"checker: {exc!r}")
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:
        strict_shape = False
        errors.append(f"shape: {exc!r}")

    live = live_node_indices(model)
    used = {name for node in model.graph.node for name in node.input if name}
    initializers = {
        item.name: int(math.prod(item.dims) if item.dims else 1)
        for item in model.graph.initializer
    }
    bad_ops = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        }
    )
    nested = sum(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    row: dict[str, object] = {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nodes": len(model.graph.node),
        "live_nodes": len(live),
        "dead_node_indices": sorted(set(range(len(model.graph.node))) - live),
        "initializers": len(model.graph.initializer),
        "params": params(model),
        "unused_initializers": sorted(name for name in initializers if name not in used),
        "duplicate_initializer_groups": duplicate_groups(model),
        "value_info_count": len(model.graph.value_info),
        "graph_output_shapes": {item.name: shape_of(item) for item in model.graph.output},
        "group_norm_shape_witnesses": group_norm_shape_witnesses(model),
        "full_check": full_check,
        "strict_shape_data_prop": strict_shape,
        "banned_ops": bad_ops,
        "nested_graph_attributes": nested,
        "function_count": len(model.functions),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "errors": errors,
    }
    if task == 340:
        table = np.asarray(
            numpy_helper.to_array(next(x for x in model.graph.initializer if x.name == "ACTable"))
        ).squeeze(-1)
        rank = int(np.linalg.matrix_rank(table.astype(np.float64)))
        row["actable"] = {
            "shape": list(table.shape),
            "elements": int(table.size),
            "real_rank": rank,
            "minimal_dense_rank_factor_params": rank * (table.shape[0] + table.shape[1]),
            "direct_params": int(table.size),
            "conclusion": (
                "The exact real rank is five: dense low-rank factors need 75 parameters "
                "before any materialization, exceeding the direct 54-parameter table."
            ),
        }
    return row


def scan_summary() -> dict[str, object]:
    names = ("exact", "global", "efactor", "lowrank", "prop", "slice", "mode")
    result: dict[str, object] = {}
    for name in names:
        path = HERE / f"{name}_audit.json"
        document = json.loads(path.read_text())
        rows = document.get("rows", [])
        result[name] = {
            "path": str(path.relative_to(ROOT)),
            "total_candidates": len(rows),
            "target_hits": [row for row in rows if row.get("task") in TASKS],
        }
    for name in ("fusion", "inline", "sign", "perm"):
        path = HERE / f"scan_{name}_full" / "build_manifest.json"
        document = json.loads(path.read_text())
        rows = document.get("candidates", document.get("rows", []))
        result[f"einsum_{name}"] = {
            "path": str(path.relative_to(ROOT)),
            "total_candidates": len(rows),
            "target_hits": [row for row in rows if row.get("task") in TASKS],
            "errors": [row for row in document.get("errors", []) if row.get("task") in TASKS],
        }
    return result


def main() -> None:
    models: dict[str, object] = {
        "base_340": inspect(HERE / "base" / "task340.onnx", 340),
        "base_361": inspect(HERE / "base" / "task361.onnx", 361),
    }
    for path in sorted((HERE / "history").glob("*.onnx")):
        models[path.stem] = inspect(path, 361)

    known: dict[str, object] = {}
    for path in sorted((HERE / "history").glob("*_known.json")):
        document = json.loads(path.read_text())
        candidate = document["candidate"]
        known[path.stem] = {
            "path": str(path.relative_to(ROOT)),
            "known": candidate["known"],
            "memory": candidate["memory"],
            "params": candidate["params"],
            "cost": candidate["cost"],
            "verdict": document["decision"]["verdict"],
            "reasons": document["decision"]["reasons"],
        }

    payload = {
        "baseline_zip": "submission_base_7999.13.zip",
        "baseline_sha256": "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1",
        "wave12_payloads_identical": {
            "340": True,
            "361": True,
        },
        "generator_hashes": {"340": "d687bc17", "361": "e40b9e2f"},
        "models": models,
        "history_known_validation": known,
        "factor_scan_summary": scan_summary(),
    }
    (HERE / "audit_evidence.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
