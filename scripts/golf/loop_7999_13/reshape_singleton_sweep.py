#!/usr/bin/env python3
"""Replace constant Reshape nodes with exact singleton-axis operators.

This is a discovery pass only.  A Reshape is changed when its fully-static
input and constant target prove that the operation merely inserts/removes
singleton axes, or is exactly a Flatten.  The produced lower-cost models must
still pass the repository's known/fresh/default-ORT/structure gates before
promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.golf.rank_dir import cost_of  # noqa: E402


def static_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    result: dict[str, tuple[int, ...]] = {}
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value in values:
        if not value.type.HasField("tensor_type"):
            continue
        dims = value.type.tensor_type.shape.dim
        if all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims):
            result[value.name] = tuple(int(dim.dim_value) for dim in dims)
    for item in inferred.graph.initializer:
        result[item.name] = tuple(int(dim) for dim in item.dims)
    return result


def insertion_axes(source: tuple[int, ...], target: tuple[int, ...]) -> tuple[int, ...] | None:
    count = len(target) - len(source)
    if count <= 0:
        return None
    singleton_positions = [index for index, dim in enumerate(target) if dim == 1]
    for axes in itertools.combinations(singleton_positions, count):
        removed = set(axes)
        if tuple(dim for index, dim in enumerate(target) if index not in removed) == source:
            return tuple(axes)
    return None


def removal_axes(source: tuple[int, ...], target: tuple[int, ...]) -> tuple[int, ...] | None:
    count = len(source) - len(target)
    if count <= 0:
        return None
    singleton_positions = [index for index, dim in enumerate(source) if dim == 1]
    for axes in itertools.combinations(singleton_positions, count):
        removed = set(axes)
        if tuple(dim for index, dim in enumerate(source) if index not in removed) == target:
            return tuple(axes)
    return None


def flatten_axis(source: tuple[int, ...], target: tuple[int, ...]) -> int | None:
    if len(target) != 2:
        return None
    for axis in range(len(source) + 1):
        left = math.prod(source[:axis])
        right = math.prod(source[axis:])
        if (left, right) == target:
            return axis
    return None


def replace_node(node: onnx.NodeProto, op_type: str, axes: tuple[int, ...] | None, opset: int) -> None:
    data_input = node.input[0]
    shape_input = node.input[1]
    del node.attribute[:]
    node.op_type = op_type
    if op_type == "Flatten":
        del node.input[:]
        node.input.extend([data_input])
        node.attribute.extend([helper.make_attribute("axis", int(axes[0]))])
    elif opset >= 13:
        del node.input[:]
        node.input.extend([data_input, shape_input])
    else:
        del node.input[:]
        node.input.extend([data_input])
        node.attribute.extend([helper.make_attribute("axes", [int(axis) for axis in axes])])


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    candidate = onnx.ModelProto()
    candidate.CopyFrom(model)
    shapes = static_shapes(candidate)
    initializers = {item.name: item for item in candidate.graph.initializer}
    use_count: dict[str, int] = {}
    for node in candidate.graph.node:
        for name in node.input:
            use_count[name] = use_count.get(name, 0) + 1
    opset = max(
        (int(item.version) for item in candidate.opset_import if item.domain in ("", "ai.onnx")),
        default=0,
    )
    changes: list[dict[str, object]] = []
    remove_initializers: set[str] = set()

    for index, node in enumerate(candidate.graph.node):
        if node.op_type != "Reshape" or len(node.input) != 2 or len(node.output) != 1:
            continue
        data_name, shape_name = node.input
        shape_init = initializers.get(shape_name)
        if shape_init is None or use_count.get(shape_name) != 1:
            continue
        target_array = np.asarray(numpy_helper.to_array(shape_init)).reshape(-1)
        if target_array.size == 0 or np.any(target_array <= 0):
            continue
        source = shapes.get(data_name)
        target = tuple(int(value) for value in target_array)
        if source is None or shapes.get(node.output[0]) not in (None, target):
            continue

        op_type: str | None = None
        axes: tuple[int, ...] | None = insertion_axes(source, target)
        if axes is not None:
            op_type = "Unsqueeze"
        else:
            axes = removal_axes(source, target)
            if axes is not None:
                op_type = "Squeeze"
        if op_type is None:
            axis = flatten_axis(source, target)
            if axis is not None:
                op_type = "Flatten"
                axes = (axis,)
        if op_type is None or axes is None:
            continue

        old_params = int(target_array.size)
        new_params = len(axes) if op_type != "Flatten" and opset >= 13 else 0
        if new_params >= old_params:
            continue
        replace_node(node, op_type, axes, opset)
        if op_type != "Flatten" and opset >= 13:
            shape_init.CopyFrom(
                numpy_helper.from_array(np.asarray(axes, dtype=np.int64), name=shape_name)
            )
        else:
            remove_initializers.add(shape_name)
        changes.append(
            {
                "node_index": index,
                "node_name": node.name,
                "data": data_name,
                "output": node.output[0],
                "source_shape": list(source),
                "target_shape": list(target),
                "replacement": op_type,
                "axes": list(axes),
                "parameter_reduction": old_params - new_params,
            }
        )

    if remove_initializers:
        kept = [item for item in candidate.graph.initializer if item.name not in remove_initializers]
        del candidate.graph.initializer[:]
        candidate.graph.initializer.extend(kept)
    return candidate, changes


def measure(model: onnx.ModelProto, task: int) -> int:
    with tempfile.TemporaryDirectory(prefix=f"ngolf_reshape_{task:03d}_") as directory:
        path = Path(directory) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        return int(cost_of(str(path))[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    costs = json.loads(args.base_costs.read_text())["costs"]
    winners: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with zipfile.ZipFile(args.baseline) as archive:
        for task in range(1, 401):
            try:
                original = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
                candidate, changes = transform(original)
                if not changes:
                    continue
                onnx.checker.check_model(candidate, full_check=True)
                onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                entry = costs.get(str(task))
                base_cost = int(entry["cost"] if isinstance(entry, dict) else entry)
                candidate_cost = measure(candidate, task)
                if candidate_cost >= base_cost:
                    continue
                path = args.out_dir / f"task{task:03d}.onnx"
                onnx.save(candidate, path)
                winners.append(
                    {
                        "task": task,
                        "path": str(path),
                        "baseline_cost": base_cost,
                        "candidate_cost": candidate_cost,
                        "projected_gain": math.log(base_cost / candidate_cost),
                        "changes": changes,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
                print(f"task{task:03d}: {base_cost}->{candidate_cost} {changes}")
            except Exception as exc:  # noqa: BLE001
                failures.append({"task": task, "error": repr(exc)})

    payload = {
        "baseline": str(args.baseline),
        "winners": winners,
        "projected_gain": sum(float(item["projected_gain"]) for item in winners),
        "failures": failures,
    }
    (args.out_dir / "manifest_pre_differential.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"winners": len(winners), "gain": payload["projected_gain"], "failures": len(failures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
