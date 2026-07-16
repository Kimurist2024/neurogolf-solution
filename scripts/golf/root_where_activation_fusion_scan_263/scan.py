#!/usr/bin/env python3
"""Fuse exact comparison/Where activation patterns across all 400 tasks."""

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
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"
FLOAT_TYPES = {TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.BFLOAT16}


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"whereactivation263_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def metadata(model: onnx.ModelProto) -> tuple[dict[str, int], dict[str, tuple[int, ...] | None]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    types: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...] | None] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor = value.type.tensor_type
        types[value.name] = tensor.elem_type
        dims = tensor.shape.dim
        shapes[value.name] = (
            tuple(int(dim.dim_value) for dim in dims)
            if all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims)
            else None
        )
    for init in inferred.graph.initializer:
        types[init.name] = init.data_type
        shapes[init.name] = tuple(int(dim) for dim in init.dims)
    return types, shapes


def scalar(array: np.ndarray | None) -> float | None:
    if array is None or array.size != 1 or array.dtype.kind not in "fiu":
        return None
    value = float(array.reshape(-1)[0])
    return value if math.isfinite(value) else None


def all_zero(array: np.ndarray | None) -> bool:
    return bool(array is not None and array.size and np.all(array == 0))


def attrs(node: onnx.NodeProto) -> dict:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def prune(model: onnx.ModelProto) -> tuple[list[str], list[str]]:
    graph_outputs = {value.name for value in model.graph.output}
    removed_nodes: list[str] = []
    while True:
        uses = Counter(name for node in model.graph.node for name in node.input if name)
        keep = []
        current = []
        for node in model.graph.node:
            outputs = [name for name in node.output if name]
            if outputs and all(uses[name] == 0 and name not in graph_outputs for name in outputs):
                current.append(node.name or node.op_type)
            else:
                keep.append(node)
        if not current:
            break
        removed_nodes.extend(current)
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


def pattern(
    node: onnx.NodeProto,
    producer: dict[str, tuple[int, onnx.NodeProto]],
    arrays: dict[str, np.ndarray],
    types: dict[str, int],
    shapes: dict[str, tuple[int, ...] | None],
) -> tuple[str, str, float, list[int]] | None:
    if node.op_type != "Where" or len(node.input) != 3:
        return None
    cond_ref = producer.get(node.input[0])
    if cond_ref is None:
        return None
    cond_index, cond = cond_ref
    if cond.op_type != "Greater" or len(cond.input) != 2:
        return None
    x = cond.input[0]
    threshold = scalar(arrays.get(cond.input[1]))
    if threshold is None or types.get(x) not in FLOAT_TYPES:
        return None
    if node.input[1] != x:
        return None
    if shapes.get(x) is None or shapes.get(x) != shapes.get(node.output[0]):
        return None
    if all_zero(arrays.get(node.input[2])):
        if threshold == 0.0:
            return "Relu", x, threshold, [cond_index]
        return "ThresholdedRelu", x, threshold, [cond_index]
    # Where(x > 0, x, alpha*x) is LeakyRelu(x, alpha).
    negative_ref = producer.get(node.input[2])
    if threshold != 0.0 or negative_ref is None:
        return None
    mul_index, mul = negative_ref
    if mul.op_type != "Mul" or len(mul.input) != 2:
        return None
    for const_index in (0, 1):
        if mul.input[1 - const_index] != x:
            continue
        slope = scalar(arrays.get(mul.input[const_index]))
        if slope is not None:
            return "LeakyRelu", x, slope, [cond_index, mul_index]
    return None


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    census = {"Where": 0, "Greater": 0, "patterns": 0}
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            producer = {
                output: (index, node)
                for index, node in enumerate(model.graph.node)
                for output in node.output if output
            }
            try:
                types, shapes = metadata(model)
            except Exception:
                types, shapes = {}, {}
            baseline = None
            for index, node in enumerate(model.graph.node):
                census["Where"] += node.op_type == "Where"
                census["Greater"] += node.op_type == "Greater"
                found = pattern(node, producer, arrays, types, shapes)
                if found is None:
                    continue
                census["patterns"] += 1
                op_type, x, alpha, upstream = found
                candidate = copy.deepcopy(model)
                original = candidate.graph.node[index]
                if op_type == "Relu":
                    replacement = helper.make_node("Relu", [x], list(original.output), name=original.name)
                elif op_type == "ThresholdedRelu":
                    replacement = helper.make_node(
                        "ThresholdedRelu", [x], list(original.output), name=original.name, alpha=alpha
                    )
                else:
                    replacement = helper.make_node(
                        "LeakyRelu", [x], list(original.output), name=original.name, alpha=alpha
                    )
                original.CopyFrom(replacement)
                removed_nodes, removed_inits = prune(candidate)
                record: dict = {
                    "task": task,
                    "node_index": index,
                    "upstream_indices": upstream,
                    "rewrite": op_type,
                    "alpha": alpha,
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
