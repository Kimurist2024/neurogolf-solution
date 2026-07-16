#!/usr/bin/env python3
"""Factor exact rank-one Einsum initializer tensors into axis vectors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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


def outer(vectors: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    result = np.asarray(vectors[0], dtype=dtype)
    for vector in vectors[1:]:
        result = np.multiply(
            result.reshape(result.shape + (1,)),
            np.asarray(vector, dtype=dtype).reshape((1,) * result.ndim + vector.shape),
            dtype=dtype,
        )
    return np.asarray(result, dtype=dtype)


def exact_vectors(array: np.ndarray) -> list[np.ndarray] | None:
    if array.ndim < 2 or array.size == 0 or not np.any(array != 0):
        return None
    if array.dtype.kind not in "biufc":
        return None
    pivot_flat = int(np.flatnonzero(array != 0)[0])
    pivot_index = tuple(int(index) for index in np.unravel_index(pivot_flat, array.shape))
    pivot = array[pivot_index]
    vectors: list[np.ndarray] = []
    for axis in range(array.ndim):
        selection: list[int | slice] = list(pivot_index)
        selection[axis] = slice(None)
        vector = np.asarray(array[tuple(selection)])
        if axis == 0:
            candidate = vector.astype(array.dtype, copy=True)
        else:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                candidate = np.divide(vector, pivot).astype(array.dtype)
        vectors.append(candidate)
    try:
        rebuilt = outer(vectors, array.dtype)
    except Exception:
        return None
    if not np.array_equal(rebuilt, array, equal_nan=True):
        return None
    return vectors


def plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    _, locations = uses(model.graph)
    result: list[dict[str, object]] = []
    for name, array in arrays.items():
        if array.ndim < 2 or sum(array.shape) >= array.size or not locations[name]:
            continue
        valid = True
        for node_index, input_index in locations[name]:
            node = model.graph.node[node_index]
            if node.op_type != "Einsum":
                valid = False
                break
            text = equation_attribute(node).s.decode("ascii")
            operands = text.split("->", 1)[0].split(",")
            if (
                input_index >= len(operands)
                or "..." in operands[input_index]
                or len(operands[input_index]) != array.ndim
                or len(node.input) + array.ndim - 1 >= 15
            ):
                valid = False
                break
        if not valid:
            continue
        vectors = exact_vectors(array)
        if vectors is None:
            continue
        result.append(
            {
                "initializer": name,
                "shape": list(array.shape),
                "original_params": int(array.size),
                "factor_params": int(sum(vector.size for vector in vectors)),
                "parameter_saving": int(array.size - sum(vector.size for vector in vectors)),
                "vectors": vectors,
            }
        )
    return result


def build(source: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    name = str(plan["initializer"])
    vectors = plan["vectors"]
    if not isinstance(vectors, list) or not all(isinstance(item, np.ndarray) for item in vectors):
        raise TypeError("invalid vectors")

    # Alias identical factors, including pre-existing initializers, to avoid
    # paying twice for repeated one/selector vectors.
    canonical: dict[tuple[str, tuple[int, ...], bytes], str] = {}
    for item in model.graph.initializer:
        if item.name == name:
            continue
        array = np.ascontiguousarray(numpy_helper.to_array(item))
        canonical[(array.dtype.str, tuple(array.shape), array.tobytes())] = item.name
    factor_names: list[str] = []
    for axis, vector in enumerate(vectors):
        value = np.ascontiguousarray(vector)
        key = (value.dtype.str, tuple(value.shape), value.tobytes())
        factor_name = canonical.get(key)
        if factor_name is None:
            factor_name = f"{name}__outer_axis{axis}"
            model.graph.initializer.append(numpy_helper.from_array(value, factor_name))
            canonical[key] = factor_name
        factor_names.append(factor_name)

    replacements = 0
    for node in model.graph.node:
        positions = [index for index, input_name in enumerate(node.input) if input_name == name]
        if not positions:
            continue
        attr = equation_attribute(node)
        text = attr.s.decode("ascii")
        lhs, rhs = text.split("->", 1)
        terms = lhs.split(",")
        inputs = list(node.input)
        for position in reversed(positions):
            term = terms[position]
            inputs[position : position + 1] = factor_names
            terms[position : position + 1] = list(term)
            replacements += 1
        del node.input[:]
        node.input.extend(inputs)
        attr.s = (",".join(terms) + "->" + rhs).encode("ascii")
    if replacements == 0:
        raise RuntimeError("initializer had no uses")
    remaining = Counter(input_name for node in model.graph.node for input_name in node.input if input_name)
    if remaining[name]:
        raise RuntimeError("initializer remains used")
    kept = [item for item in model.graph.initializer if item.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"outer_factor_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        return int(memory), int(params), int(cost)


def concise(plan: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in plan.items() if key != "vectors"}


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
                info = concise(plan)
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
                            **info,
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
                    errors.append({"task": task, "stage": "build", **info, "error": repr(exc)})
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
