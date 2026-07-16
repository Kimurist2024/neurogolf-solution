#!/usr/bin/env python3
"""Audit and apply input-independent exact ONNX graph rewrites.

Only two rewrite classes are emitted:

* TensorProto-identical initializer aliasing (ignoring the initializer name).
* Unary identity nodes whose inferred input/output dtype and shape are equal:
  Identity, no-op Cast, no-op Reshape, and identity-permutation Transpose.

The pass deliberately avoids algebra that depends on floating-point edge cases.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


def tensor_key(tensor: onnx.TensorProto) -> bytes:
    clone = onnx.TensorProto()
    clone.CopyFrom(tensor)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def dims(value: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or int(dim.dim_value) <= 0:
            return None
        result.append(int(dim.dim_value))
    return tuple(result)


def type_map(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int, ...] | None]]:
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    result: dict[str, tuple[int, tuple[int, ...] | None]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if value.type.HasField("tensor_type"):
            result[value.name] = (int(value.type.tensor_type.elem_type), dims(value))
    for tensor in inferred.graph.initializer:
        result[tensor.name] = (int(tensor.data_type), tuple(int(item) for item in tensor.dims))
    return result


def consumers(model: onnx.ModelProto) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                result[name].append((node_index, input_index))
    return result


def replace_all_inputs(model: onnx.ModelProto, old: str, new: str) -> int:
    count = 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
                count += 1
    return count


def attr_ints(node: onnx.NodeProto, name: str) -> tuple[int, ...] | None:
    for attribute in node.attribute:
        if attribute.name == name:
            return tuple(int(item) for item in attribute.ints)
    return None


def initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}


def exact_unary_aliases(model: onnx.ModelProto) -> list[dict[str, object]]:
    types = type_map(model)
    arrays = initializer_arrays(model)
    graph_outputs = {value.name for value in model.graph.output}
    aliases: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        if len(node.input) < 1 or len(node.output) != 1 or not node.input[0] or not node.output[0]:
            continue
        source, target = node.input[0], node.output[0]
        if target in graph_outputs or types.get(source) != types.get(target):
            continue
        proof = None
        if node.op_type == "Identity":
            proof = "Identity(x)=x"
        elif node.op_type == "Cast":
            proof = "Cast to the already inferred input dtype"
        elif node.op_type == "Transpose":
            rank = len(types[source][1] or ())
            perm = attr_ints(node, "perm")
            if perm is None:
                perm = tuple(reversed(range(rank)))
            if perm == tuple(range(rank)):
                proof = "identity permutation"
        elif node.op_type == "Reshape" and len(node.input) >= 2 and node.input[1] in arrays:
            requested = tuple(int(item) for item in arrays[node.input[1]].reshape(-1))
            shape = types[source][1]
            if shape is not None and requested == shape:
                proof = "requested static shape equals inferred input shape"
        if proof is not None:
            aliases.append({"index": index, "op": node.op_type, "input": source, "output": target, "proof": proof})
    return aliases


def duplicate_groups(model: onnx.ModelProto) -> list[dict[str, object]]:
    groups: dict[bytes, list[onnx.TensorProto]] = defaultdict(list)
    for tensor in model.graph.initializer:
        groups[tensor_key(tensor)].append(tensor)
    result = []
    for tensors in groups.values():
        if len(tensors) < 2:
            continue
        result.append(
            {
                "canonical": tensors[0].name,
                "duplicates": [tensor.name for tensor in tensors[1:]],
                "saved_params": sum(math.prod(tensor.dims) if tensor.dims else 1 for tensor in tensors[1:]),
            }
        )
    return result


def rewrite(source: Path, output: Path) -> dict[str, object]:
    original = onnx.load(source)
    candidate = copy.deepcopy(original)
    duplicate_plan = duplicate_groups(candidate)
    removed_initializers: set[str] = set()
    for group in duplicate_plan:
        for duplicate in group["duplicates"]:
            replace_all_inputs(candidate, str(duplicate), str(group["canonical"]))
            removed_initializers.add(str(duplicate))
    kept = [tensor for tensor in candidate.graph.initializer if tensor.name not in removed_initializers]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)

    unary_plan = exact_unary_aliases(candidate)
    remove_nodes: set[int] = set()
    for row in unary_plan:
        replace_all_inputs(candidate, str(row["output"]), str(row["input"]))
        remove_nodes.add(int(row["index"]))
    kept_nodes = [node for index, node in enumerate(candidate.graph.node) if index not in remove_nodes]
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept_nodes)

    onnx.checker.check_model(candidate, full_check=True)
    shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(candidate, output)
    before = cost_of(str(source))
    after = cost_of(str(output))
    return {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "cost_before": {"memory": before[0], "params": before[1], "cost": before[2]},
        "cost_after": {"memory": after[0], "params": after[1], "cost": after[2]},
        "duplicate_initializer_aliases": duplicate_plan,
        "unary_aliases": unary_plan,
        "checker_full": True,
        "strict_data_prop": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = rewrite(args.source.resolve(), args.output.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
