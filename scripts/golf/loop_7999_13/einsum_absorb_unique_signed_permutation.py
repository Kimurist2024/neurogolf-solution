#!/usr/bin/env python3
"""Absorb a unique signed-permutation Einsum operand into a unique constant."""

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


def equation(node: onnx.NodeProto) -> str:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode()
    raise ValueError("Einsum equation missing")


def set_equation(node: onnx.NodeProto, terms: list[str], output: str) -> None:
    value = ",".join(terms) + "->" + output
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = value.encode()
            return
    raise ValueError("Einsum equation missing")


def locations(graph: onnx.GraphProto) -> tuple[Counter[str], dict[str, list[tuple[int, int]]]]:
    counts: Counter[str] = Counter()
    result: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                counts[name] += 1
                result[name].append((node_index, input_index))
    return counts, result


def is_signed_permutation(array: np.ndarray) -> bool:
    return bool(
        array.ndim == 2
        and array.shape[0] == array.shape[1]
        and array.shape[0] > 1
        and array.dtype.kind in "iufc"
        and np.all((array == 0) | (array == 1) | (array == -1))
        and np.all(np.count_nonzero(array, axis=0) == 1)
        and np.all(np.count_nonzero(array, axis=1) == 1)
    )


def transform_axis(
    target: np.ndarray,
    target_axis: int,
    permutation: np.ndarray,
    contracted_position: int,
) -> np.ndarray:
    result = np.tensordot(
        target,
        permutation,
        axes=([target_axis], [contracted_position]),
    )
    result = np.moveaxis(result, -1, target_axis)
    return result.astype(target.dtype, copy=False)


def plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    initializers = {item.name: item for item in model.graph.initializer}
    counts, use_locations = locations(model.graph)
    result: list[dict[str, object]] = []
    for source_name, source_proto in initializers.items():
        if counts[source_name] != 1:
            continue
        source = np.asarray(numpy_helper.to_array(source_proto))
        if not is_signed_permutation(source):
            continue
        node_index, source_index = use_locations[source_name][0]
        node = model.graph.node[node_index]
        if node.op_type != "Einsum" or "->" not in equation(node) or "..." in equation(node):
            continue
        lhs, output = equation(node).split("->", 1)
        terms = lhs.split(",")
        if len(terms) != len(node.input):
            continue
        source_term = terms[source_index]
        if len(source_term) != 2 or source_term[0] == source_term[1]:
            continue
        for contracted_position in (0, 1):
            contracted = source_term[contracted_position]
            replacement = source_term[1 - contracted_position]
            if contracted in output:
                continue
            occurrences: list[tuple[int, int]] = []
            for input_index, term in enumerate(terms):
                if input_index == source_index:
                    continue
                occurrences.extend(
                    (input_index, axis)
                    for axis, label in enumerate(term)
                    if label == contracted
                )
            if len(occurrences) != 1:
                continue
            target_index, target_axis = occurrences[0]
            target_name = node.input[target_index]
            target_proto = initializers.get(target_name)
            if target_proto is None or counts[target_name] != 1:
                continue
            target = np.asarray(numpy_helper.to_array(target_proto))
            target_term = terms[target_index]
            if (
                target.dtype != source.dtype
                or len(target_term) != target.ndim
                or replacement in target_term
                or target.shape[target_axis] != source.shape[contracted_position]
            ):
                continue
            product = transform_axis(target, target_axis, source, contracted_position)
            result.append(
                {
                    "source": source_name,
                    "source_elements": int(source.size),
                    "node_index": node_index,
                    "source_index": source_index,
                    "source_term": source_term,
                    "contracted_position": contracted_position,
                    "contracted_label": contracted,
                    "replacement_label": replacement,
                    "target": target_name,
                    "target_index": target_index,
                    "target_axis": target_axis,
                    "target_term": target_term,
                    "product": product,
                }
            )
    return result


def build(source_model: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    model = copy.deepcopy(source_model)
    graph = model.graph
    source_name = str(plan["source"])
    target_name = str(plan["target"])
    product = plan["product"]
    if not isinstance(product, np.ndarray):
        raise TypeError("missing product")
    kept: list[onnx.TensorProto] = []
    for item in graph.initializer:
        if item.name == source_name:
            continue
        if item.name == target_name:
            kept.append(numpy_helper.from_array(product, target_name))
        else:
            kept.append(item)
    del graph.initializer[:]
    graph.initializer.extend(kept)

    node = graph.node[int(plan["node_index"])]
    lhs, output = equation(node).split("->", 1)
    terms = lhs.split(",")
    target_index = int(plan["target_index"])
    target_axis = int(plan["target_axis"])
    old_target_term = terms[target_index]
    terms[target_index] = (
        old_target_term[:target_axis]
        + str(plan["replacement_label"])
        + old_target_term[target_axis + 1 :]
    )
    source_index = int(plan["source_index"])
    inputs = [name for index, name in enumerate(node.input) if index != source_index]
    terms = [term for index, term in enumerate(terms) if index != source_index]
    del node.input[:]
    node.input.extend(inputs)
    set_equation(node, terms, output)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"unique_perm_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        return int(memory), int(params), int(cost)


def concise(plan: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in plan.items() if key != "product"}


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
