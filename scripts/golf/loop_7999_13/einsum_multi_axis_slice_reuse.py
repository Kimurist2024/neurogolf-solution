#!/usr/bin/env python3
"""Reuse an Einsum initializer that is an exact multi-axis slice of another.

The existing one-axis pass misses tensors obtained by fixing two or more axes
of a stored higher-rank tensor.  Each fixed axis is selected by a one-hot
vector contracted inside the existing Einsum, so no runtime tensor is added.
Candidates that would create or enlarge a giant Einsum are refused.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import string
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def uses(graph: onnx.GraphProto) -> tuple[Counter[str], dict[str, list[tuple[int, int]]]]:
    counts: Counter[str] = Counter()
    locations: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                counts[name] += 1
                locations[name].append((node_index, input_index))
    return counts, locations


def plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    _, locations = uses(model.graph)
    eligible: list[str] = []
    for name, array in arrays.items():
        if array.ndim == 0 or not locations[name]:
            continue
        valid = True
        for node_index, input_index in locations[name]:
            node = model.graph.node[node_index]
            if node.op_type != "Einsum" or len(node.input) >= 14:
                valid = False
                break
            text = equation_attribute(node).s.decode("ascii")
            operands = text.split("->", 1)[0].split(",")
            if (
                input_index >= len(operands)
                or "..." in operands[input_index]
                or len(operands[input_index]) != array.ndim
            ):
                valid = False
                break
        if valid:
            eligible.append(name)

    result: list[dict[str, object]] = []
    for small_name in eligible:
        small = arrays[small_name]
        for large_name, large in arrays.items():
            if (
                large_name == small_name
                or large.dtype != small.dtype
                or large.ndim <= small.ndim + 1
                or large.ndim - small.ndim > 4
                or large.size <= small.size
            ):
                continue
            fixed_count = large.ndim - small.ndim
            for fixed_axes in itertools.combinations(range(large.ndim), fixed_count):
                fixed_set = set(fixed_axes)
                remaining_shape = tuple(
                    size for axis, size in enumerate(large.shape) if axis not in fixed_set
                )
                if remaining_shape != small.shape:
                    continue
                selector_params = sum(int(large.shape[axis]) for axis in fixed_axes)
                if selector_params >= small.size:
                    continue
                if any(
                    len(model.graph.node[node_index].input) + fixed_count - 1 >= 15
                    for node_index, _ in locations[small_name]
                ):
                    continue
                fixed_shape = tuple(int(large.shape[axis]) for axis in fixed_axes)
                for indices in np.ndindex(fixed_shape):
                    selection: list[int | slice] = [slice(None)] * large.ndim
                    for axis, index in zip(fixed_axes, indices):
                        selection[axis] = int(index)
                    if np.array_equal(large[tuple(selection)], small, equal_nan=True):
                        result.append(
                            {
                                "small": small_name,
                                "small_shape": list(small.shape),
                                "large": large_name,
                                "large_shape": list(large.shape),
                                "fixed_axes": list(fixed_axes),
                                "indices": [int(index) for index in indices],
                                "selector_params": selector_params,
                                "removed_params": int(small.size),
                                "parameter_saving": int(small.size - selector_params),
                            }
                        )
                        break
    return result


def build(source: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    small_name = str(plan["small"])
    large_name = str(plan["large"])
    small = arrays[small_name]
    large = arrays[large_name]
    fixed_axes = tuple(int(axis) for axis in plan["fixed_axes"])
    indices = tuple(int(index) for index in plan["indices"])

    selector_names: list[str] = []
    for ordinal, (axis, index) in enumerate(zip(fixed_axes, indices)):
        name = f"{small_name}__slice_{ordinal}__of__{large_name}"
        selector = np.zeros((large.shape[axis],), dtype=large.dtype)
        selector[index] = 1
        model.graph.initializer.append(numpy_helper.from_array(selector, name))
        selector_names.append(name)

    replaced = 0
    for node in model.graph.node:
        positions = [index for index, name in enumerate(node.input) if name == small_name]
        if not positions:
            continue
        attr = equation_attribute(node)
        text = attr.s.decode("ascii")
        lhs, rhs = text.split("->", 1)
        terms = lhs.split(",")
        inputs = list(node.input)
        used = set("".join(terms) + rhs)
        available = iter(label for label in string.ascii_letters if label not in used)
        for position in reversed(positions):
            old_term = terms[position]
            latent = {axis: next(available) for axis in fixed_axes}
            old_cursor = 0
            large_labels: list[str] = []
            for axis in range(large.ndim):
                if axis in latent:
                    large_labels.append(latent[axis])
                else:
                    large_labels.append(old_term[old_cursor])
                    old_cursor += 1
            if old_cursor != len(old_term):
                raise RuntimeError("failed to map small labels")
            replacement_inputs = [large_name, *selector_names]
            replacement_terms = [
                "".join(large_labels),
                *(latent[axis] for axis in fixed_axes),
            ]
            inputs[position : position + 1] = replacement_inputs
            terms[position : position + 1] = replacement_terms
            replaced += 1
        del node.input[:]
        node.input.extend(inputs)
        attr.s = (",".join(terms) + "->" + rhs).encode("ascii")

    if replaced == 0:
        raise RuntimeError("small initializer had no uses")
    remaining = Counter(name for node in model.graph.node for name in node.input if name)
    if remaining[small_name]:
        raise RuntimeError("small initializer still used")
    kept = [item for item in model.graph.initializer if item.name != small_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"multi_slice_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        return int(memory), int(params), int(cost)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ort.set_default_logger_severity(4)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            model = onnx.load_model_from_string(archive.read(member))
            task_plans = plans(model)
            if not task_plans:
                continue
            try:
                base_memory, base_params, base_cost = measure(model, task)
            except Exception as exc:
                errors.append({"task": task, "stage": "base_cost", "error": repr(exc)})
                continue
            for ordinal, plan in enumerate(task_plans, 1):
                try:
                    candidate = build(model, plan)
                    memory, params, cost = measure(candidate, task)
                    if cost < 0 or cost >= base_cost:
                        continue
                    path = args.output_dir / f"task{task:03d}_r{ordinal:02d}.onnx"
                    onnx.save(candidate, path)
                    rows.append(
                        {
                            "task": task,
                            "path": str(path),
                            **plan,
                            "baseline_memory": base_memory,
                            "baseline_params": base_params,
                            "baseline_cost": base_cost,
                            "candidate_memory": memory,
                            "candidate_params": params,
                            "candidate_cost": cost,
                            "projected_gain": math.log(base_cost / cost),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
                except Exception as exc:
                    errors.append({"task": task, "stage": "build", **plan, "error": repr(exc)})
    payload = {"source_zip": str(args.zip), "rows": rows, "errors": errors}
    manifest = args.output_dir / "build_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_count": len(rows),
        "projected_gain": sum(float(row["projected_gain"]) for row in rows),
        "error_count": len(errors),
        "manifest": str(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
