#!/usr/bin/env python3
"""Factor Einsum matrices through exact row/column dictionaries.

A matrix with repeated rows or columns can be represented as two smaller
matrices joined by a fresh latent Einsum axis.  Only bit-exact, parameter-
reducing factorizations on non-giant Einsum nodes are emitted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def row_dictionary(array: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = array.shape
    keys: dict[bytes, int] = {}
    indices: list[int] = []
    values: list[np.ndarray] = []
    for row in array:
        key = row.tobytes()
        if key not in keys:
            keys[key] = len(values)
            values.append(row.copy())
        indices.append(keys[key])
    width = len(values)
    if width >= rows or rows * width + width * cols >= rows * cols:
        return None
    selector = np.zeros((rows, width), dtype=array.dtype)
    selector[np.arange(rows), np.asarray(indices)] = 1
    dictionary = np.stack(values, axis=0).astype(array.dtype, copy=False)
    rebuilt = np.asarray(np.einsum("ik,kj->ij", selector, dictionary, optimize=False), dtype=array.dtype)
    return (selector, dictionary) if np.array_equal(rebuilt, array, equal_nan=True) else None


def factorizations(array: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if array.ndim != 2 or array.dtype.kind != "f" or not np.all(np.isfinite(array)):
        return []
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    rows = row_dictionary(array)
    if rows is not None:
        result.append(("rows", *rows))
    cols = row_dictionary(array.T)
    if cols is not None:
        selector_t, dictionary_t = cols
        result.append(("columns", dictionary_t.T.copy(), selector_t.T.copy()))
    return result


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
        if not locations[name]:
            continue
        valid = True
        for node_index, input_index in locations[name]:
            node = model.graph.node[node_index]
            terms = equation_attribute(node).s.decode("ascii").split("->", 1)[0].split(",") if node.op_type == "Einsum" else []
            occurrences = sum(value == name for value in node.input)
            if (
                node.op_type != "Einsum"
                or len(node.input) + occurrences > 14
                or input_index >= len(terms)
                or len(terms[input_index]) != 2
                or "..." in terms[input_index]
            ):
                valid = False
                break
        if not valid:
            continue
        for orientation, left, right in factorizations(array):
            result.append(
                {
                    "initializer": name,
                    "shape": list(array.shape),
                    "orientation": orientation,
                    "latent_size": int(left.shape[1]),
                    "removed_params": int(array.size),
                    "factor_params": int(left.size + right.size),
                    "parameter_saving": int(array.size - left.size - right.size),
                    "use_count": len(locations[name]),
                    "left": left,
                    "right": right,
                }
            )
    return result


def build(source: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    name = str(plan["initializer"])
    left_name = f"{name}__dict_left"
    right_name = f"{name}__dict_right"
    left = np.asarray(plan["left"])
    right = np.asarray(plan["right"])
    replacements = 0
    for node in model.graph.node:
        positions = [index for index, value in enumerate(node.input) if value == name]
        if not positions:
            continue
        attr = equation_attribute(node)
        lhs, rhs = attr.s.decode("ascii").split("->", 1)
        terms = lhs.split(",")
        inputs = list(node.input)
        used = set("".join(terms) + rhs)
        available = iter(label for label in string.ascii_letters if label not in used)
        for position in reversed(positions):
            term = terms[position]
            latent = next(available)
            inputs[position : position + 1] = [left_name, right_name]
            terms[position : position + 1] = [term[0] + latent, latent + term[1]]
            replacements += 1
        del node.input[:]
        node.input.extend(inputs)
        attr.s = (",".join(terms) + "->" + rhs).encode("ascii")
    if replacements == 0:
        raise RuntimeError("initializer has no replaceable uses")
    remaining = Counter(value for node in model.graph.node for value in node.input if value)
    if remaining[name]:
        raise RuntimeError("initializer remains used")
    kept = [item for item in model.graph.initializer if item.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(left, left_name))
    model.graph.initializer.append(numpy_helper.from_array(right, right_name))
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"dictionary_factor_{task:03d}_") as tmp:
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
                serializable = {key: value for key, value in plan.items() if key not in {"left", "right"}}
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
