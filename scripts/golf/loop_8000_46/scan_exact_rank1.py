#!/usr/bin/env python3
"""Find bit-exact rank-1 Einsum initializer reductions.

Only floating initializers used exclusively by small Einsum nodes are
considered.  The stored tensor must equal the dtype-rounded outer product
exactly; approximate SVD/CP factorizations are deliberately excluded.
"""

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
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def exact_factors(array: np.ndarray) -> list[np.ndarray] | None:
    if not 2 <= array.ndim <= 6 or array.dtype.kind != "f" or not np.all(np.isfinite(array)):
        return None
    if array.size <= sum(array.shape) or not np.any(array):
        return None
    coordinates = np.argwhere(array != 0)
    for coordinate_array in coordinates[:256]:
        coordinate = tuple(int(value) for value in coordinate_array)
        pivot = array[coordinate]
        for unnormalized_axis in range(array.ndim):
            factors: list[np.ndarray] = []
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                for axis in range(array.ndim):
                    selection: list[int | slice] = list(coordinate)
                    selection[axis] = slice(None)
                    values = np.asarray(array[tuple(selection)], dtype=array.dtype)
                    if axis != unnormalized_axis:
                        values = np.asarray(values / pivot, dtype=array.dtype)
                    factors.append(values)
                shape = [1] * array.ndim
                shape[0] = array.shape[0]
                rebuilt = factors[0].reshape(shape)
                for axis in range(1, array.ndim):
                    shape = [1] * array.ndim
                    shape[axis] = array.shape[axis]
                    rebuilt = np.asarray(rebuilt * factors[axis].reshape(shape), dtype=array.dtype)
            if np.array_equal(rebuilt, array, equal_nan=True):
                return factors
    return None


def opportunities(model: onnx.ModelProto) -> list[dict[str, object]]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    locations: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name in arrays:
                locations[name].append((node_index, input_index))

    result: list[dict[str, object]] = []
    for name, array in arrays.items():
        factors = exact_factors(array)
        if factors is None or not locations[name]:
            continue
        valid = True
        for node_index, input_index in locations[name]:
            node = model.graph.node[node_index]
            if node.op_type != "Einsum" or len(node.input) + array.ndim - 1 > 14:
                valid = False
                break
            terms = equation_attribute(node).s.decode("ascii").split("->", 1)[0].split(",")
            if input_index >= len(terms) or len(terms[input_index]) != array.ndim or "..." in terms[input_index]:
                valid = False
                break
        if not valid:
            continue
        result.append(
            {
                "initializer": name,
                "shape": list(array.shape),
                "removed_params": int(array.size),
                "factor_params": int(sum(factor.size for factor in factors)),
                "parameter_saving": int(array.size - sum(factor.size for factor in factors)),
                "use_count": len(locations[name]),
                "factors": factors,
            }
        )
    return result


def build(source: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    name = str(plan["initializer"])
    factors = [np.asarray(factor) for factor in plan["factors"]]
    factor_names = [f"{name}__exact_rank1_axis{axis}" for axis in range(len(factors))]
    replacements = 0
    for node in model.graph.node:
        positions = [index for index, value in enumerate(node.input) if value == name]
        if not positions:
            continue
        attr = equation_attribute(node)
        lhs, rhs = attr.s.decode("ascii").split("->", 1)
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
        raise RuntimeError("initializer has no replaceable use")
    remaining = Counter(value for node in model.graph.node for value in node.input if value)
    if remaining[name]:
        raise RuntimeError("initializer remains used")
    kept = [item for item in model.graph.initializer if item.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    for factor, factor_name in zip(factors, factor_names):
        model.graph.initializer.append(numpy_helper.from_array(factor, factor_name))
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"exact_rank1_{task:03d}_") as tmp:
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
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            for ordinal, plan in enumerate(opportunities(model), 1):
                serializable = {key: value for key, value in plan.items() if key != "factors"}
                try:
                    baseline_memory, baseline_params, baseline_cost = measure(model, task)
                    candidate = build(model, plan)
                    memory, params, cost = measure(candidate, task)
                    if cost <= 0 or cost >= baseline_cost:
                        continue
                    path = args.output_dir / f"task{task:03d}_r{ordinal:02d}.onnx"
                    onnx.save(candidate, path)
                    rows.append(
                        {
                            "task": task,
                            "path": str(path),
                            **serializable,
                            "baseline_memory": baseline_memory,
                            "baseline_params": baseline_params,
                            "baseline_cost": baseline_cost,
                            "candidate_memory": memory,
                            "candidate_params": params,
                            "candidate_cost": cost,
                            "projected_gain": math.log(baseline_cost / cost),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
                except Exception as exc:
                    errors.append({"task": task, **serializable, "error": repr(exc)})
    rows.sort(key=lambda row: (-float(row["projected_gain"]), int(row["task"])))
    manifest = args.output_dir / "build_manifest.json"
    manifest.write_text(json.dumps({"source_zip": str(args.zip), "rows": rows, "errors": errors}, indent=2) + "\n")
    print(json.dumps({
        "candidate_count": len(rows),
        "task_count": len({int(row["task"]) for row in rows}),
        "projected_gain": sum(float(row["projected_gain"]) for row in rows),
        "errors": len(errors),
        "manifest": str(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
