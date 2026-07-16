#!/usr/bin/env python3
"""Reuse an existing initializer whose exact contraction equals another initializer."""

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
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


LABELS = string.ascii_lowercase + string.ascii_uppercase


def equation(node: onnx.NodeProto) -> str:
    return next(attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation")


def contraction_plans(target: np.ndarray, source: np.ndarray) -> list[tuple[int, ...]]:
    if not 0 <= target.ndim <= source.ndim <= 6 or target.dtype != source.dtype:
        return []
    if target.size >= source.size:
        return []
    target_labels = LABELS[: target.ndim]
    reduce_labels = LABELS[target.ndim : target.ndim + source.ndim]
    assignments: list[tuple[int, ...]] = []

    def enumerate_assignments(
        source_axis: int,
        current: list[int],
        reduction_dimensions: list[int],
    ) -> None:
        if source_axis == source.ndim:
            if all(axis in current for axis in range(target.ndim)):
                assignments.append(tuple(current))
            return
        dimension = source.shape[source_axis]
        for target_axis in range(target.ndim):
            if target.shape[target_axis] == dimension:
                current.append(target_axis)
                enumerate_assignments(source_axis + 1, current, reduction_dimensions)
                current.pop()
        for group, group_dimension in enumerate(reduction_dimensions):
            if group_dimension == dimension:
                current.append(-1 - group)
                enumerate_assignments(source_axis + 1, current, reduction_dimensions)
                current.pop()
        current.append(-1 - len(reduction_dimensions))
        reduction_dimensions.append(dimension)
        enumerate_assignments(source_axis + 1, current, reduction_dimensions)
        reduction_dimensions.pop()
        current.pop()

    enumerate_assignments(0, [], [])
    result: list[tuple[int, ...]] = []
    # A repeated negative group means a diagonal followed by reduction.
    for assignment in assignments:
        valid = True
        source_labels: list[str] = []
        for source_axis, target_axis in enumerate(assignment):
            if target_axis < 0:
                source_labels.append(reduce_labels[-1 - target_axis])
            else:
                if source.shape[source_axis] != target.shape[target_axis]:
                    valid = False
                    break
                source_labels.append(target_labels[target_axis])
        if not valid:
            continue
        text = "".join(source_labels) + "->" + target_labels
        try:
            contracted = np.einsum(text, source, optimize=False)
        except (TypeError, ValueError):
            continue
        if contracted.shape != target.shape:
            continue
        if np.array_equal(np.asarray(contracted, dtype=target.dtype), target, equal_nan=True):
            result.append(tuple(int(value) for value in assignment))
    return result


def opportunities(model: onnx.ModelProto) -> list[dict[str, object]]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    locations: dict[str, list[tuple[int, int]]] = {name: [] for name in arrays}
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name in locations:
                locations[name].append((node_index, input_index))

    result: list[dict[str, object]] = []
    for target_name, target in arrays.items():
        if target.size == 0 or not locations[target_name]:
            continue
        valid_uses = True
        for node_index, input_index in locations[target_name]:
            node = model.graph.node[node_index]
            if node.op_type != "Einsum":
                valid_uses = False
                break
            terms = equation(node).split("->", 1)[0].split(",")
            if input_index >= len(terms) or "..." in terms[input_index] or len(terms[input_index]) != target.ndim:
                valid_uses = False
                break
        if not valid_uses:
            continue
        for source_name, source in arrays.items():
            if source_name == target_name:
                continue
            for assignment in contraction_plans(target, source):
                result.append(
                    {
                        "target": target_name,
                        "target_shape": list(target.shape),
                        "source": source_name,
                        "source_shape": list(source.shape),
                        "assignment": list(assignment),
                        "parameter_saving": int(target.size),
                        "use_count": len(locations[target_name]),
                    }
                )
    return result


def build(source_model: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    model = copy.deepcopy(source_model)
    target = str(plan["target"])
    source = str(plan["source"])
    assignment = [int(value) for value in plan["assignment"]]
    replacements = 0
    for node in model.graph.node:
        positions = [index for index, name in enumerate(node.input) if name == target]
        if not positions:
            continue
        attr = next(attr for attr in node.attribute if attr.name == "equation")
        lhs, rhs = attr.s.decode("ascii").split("->", 1)
        terms = lhs.split(",")
        used = set("".join(terms) + rhs)
        available = [label for label in LABELS if label not in used]
        for position in positions:
            target_term = terms[position]
            reduction_groups = sorted({value for value in assignment if value < 0}, reverse=True)
            reduce_count = len(reduction_groups)
            if len(available) < reduce_count:
                raise ValueError("not enough unused Einsum labels")
            reduction_labels = {
                group: available[index]
                for index, group in enumerate(reduction_groups)
            }
            source_term = "".join(
                reduction_labels[target_axis] if target_axis < 0 else target_term[target_axis]
                for target_axis in assignment
            )
            del available[:reduce_count]
            node.input[position] = source
            terms[position] = source_term
            replacements += 1
        attr.s = (",".join(terms) + "->" + rhs).encode("ascii")
    if replacements == 0:
        raise ValueError("target had no replaceable uses")
    remaining = Counter(name for node in model.graph.node for name in node.input if name)
    if remaining[target]:
        raise ValueError("target remains used")
    kept = [item for item in model.graph.initializer if item.name != target]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"initializer_contraction_{task:03d}_") as tmp:
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
            plans = opportunities(model)
            if not plans:
                continue
            try:
                base_memory, base_params, base_cost = measure(model, task)
            except Exception as exc:
                errors.append({"task": task, "stage": "base", "error": repr(exc)})
                continue
            for ordinal, plan in enumerate(plans, 1):
                try:
                    candidate = build(model, plan)
                    memory, params, cost = measure(candidate, task)
                    if cost <= 0 or cost >= base_cost:
                        continue
                    path = args.output_dir / f"task{task:03d}_r{ordinal:03d}.onnx"
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
                    errors.append({"task": task, "stage": "candidate", **plan, "error": repr(exc)})
    rows.sort(key=lambda row: (-float(row["projected_gain"]), int(row["task"])))
    manifest = args.output_dir / "build_manifest.json"
    manifest.write_text(json.dumps({"source_zip": str(args.zip), "rows": rows, "errors": errors}, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "task_count": len({int(row['task']) for row in rows}), "projected_gain": sum(float(row["projected_gain"]) for row in rows), "errors": len(errors), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
