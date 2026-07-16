#!/usr/bin/env python3
"""Find full-overwrite ScatterElements nodes equivalent to their updates."""

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

import numpy as np
import onnx
from onnx import helper, numpy_helper

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def attrs(node: onnx.NodeProto) -> dict:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"scatteridentity259_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...] | None]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    result: dict[str, tuple[int, ...] | None] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        dims = value.type.tensor_type.shape.dim
        if all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims):
            result[value.name] = tuple(int(dim.dim_value) for dim in dims)
        else:
            result[value.name] = None
    for init in inferred.graph.initializer:
        result[init.name] = tuple(int(dim) for dim in init.dims)
    return result


def is_full_identity_indices(indices: np.ndarray, axis: int) -> bool:
    if indices.ndim == 0 or axis < 0 or axis >= indices.ndim:
        return False
    axis_size = indices.shape[axis]
    expected_shape = [1] * indices.ndim
    expected_shape[axis] = axis_size
    expected = np.arange(axis_size, dtype=np.int64).reshape(expected_shape)
    normalized = np.asarray(indices, dtype=np.int64)
    normalized = np.where(normalized < 0, normalized + axis_size, normalized)
    return bool(np.array_equal(normalized, np.broadcast_to(expected, indices.shape)))


def prune(model: onnx.ModelProto) -> tuple[list[str], list[str]]:
    graph_outputs = {value.name for value in model.graph.output}
    removed_nodes: list[str] = []
    while True:
        uses = Counter(name for node in model.graph.node for name in node.input if name)
        keep = []
        removed_this_round = []
        for node in model.graph.node:
            outputs = [name for name in node.output if name]
            if outputs and all(uses[name] == 0 and name not in graph_outputs for name in outputs):
                removed_this_round.append(node.name or node.op_type)
            else:
                keep.append(node)
        if not removed_this_round:
            break
        removed_nodes.extend(removed_this_round)
        del model.graph.node[:]
        model.graph.node.extend(keep)
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    graph_inputs = {value.name for value in model.graph.input}
    removed_inits = [
        item.name for item in model.graph.initializer
        if uses[item.name] == 0 and item.name not in graph_inputs
    ]
    keep_inits = [item for item in model.graph.initializer if item.name not in removed_inits]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_inits)
    return removed_nodes, removed_inits


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    census = {"ScatterElements": 0, "constant_indices": 0, "full_identity": 0}
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            try:
                shapes = shape_map(model)
            except Exception:
                shapes = {}
            baseline = None
            for index, node in enumerate(model.graph.node):
                if node.op_type != "ScatterElements" or len(node.input) < 3:
                    continue
                census["ScatterElements"] += 1
                indices = arrays.get(node.input[1])
                if indices is None:
                    continue
                census["constant_indices"] += 1
                properties = attrs(node)
                reduction = properties.get("reduction", b"none")
                if isinstance(reduction, bytes):
                    reduction = reduction.decode()
                if reduction != "none":
                    continue
                data_shape = shapes.get(node.input[0])
                updates_shape = shapes.get(node.input[2])
                output_shape = shapes.get(node.output[0])
                if data_shape is None or data_shape != tuple(indices.shape):
                    continue
                if updates_shape != data_shape or output_shape != data_shape:
                    continue
                axis = int(properties.get("axis", 0))
                if axis < 0:
                    axis += len(data_shape)
                if not is_full_identity_indices(indices, axis):
                    continue
                census["full_identity"] += 1
                candidate = copy.deepcopy(model)
                original = candidate.graph.node[index]
                original.CopyFrom(helper.make_node(
                    "Identity", [original.input[2]], list(original.output), name=original.name
                ))
                removed_nodes, removed_inits = prune(candidate)
                record: dict = {
                    "task": task,
                    "node_index": index,
                    "axis": axis,
                    "shape": list(data_shape),
                    "indices": node.input[1],
                    "index_elements": int(indices.size),
                    "removed_nodes": removed_nodes,
                    "removed_initializers": removed_inits,
                }
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    record["baseline"] = baseline
                    record["candidate"] = current
                    record["strict_lower"] = current["cost"] < baseline["cost"]
                    if record["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{index:04d}.onnx"
                        onnx.save(candidate, path)
                        record["path"] = str(path.relative_to(REPO))
                        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(record)
    result = {
        "authority": str(AUTHORITY),
        "tasks": len(members),
        "census": census,
        "strict_lower": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"census": census, "strict_lower": result["strict_lower"]}, indent=2))


if __name__ == "__main__":
    main()
