#!/usr/bin/env python3
"""Absorb a unique signed-permutation factor into a shared Einsum initializer.

The rewrite is a discrete change of basis.  A unique square signed-permutation
operand is contracted into one axis of a shared constant.  Every other use of
that shared axis is compensated by applying the same orthogonal permutation to
a unique constant at that use.  No new tensor or initializer is introduced.
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


def equation(node: onnx.NodeProto) -> str:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode()
    raise ValueError("Einsum equation missing")


def terms(node: onnx.NodeProto) -> list[str]:
    return equation(node).split("->", 1)[0].split(",")


def output_term(node: onnx.NodeProto) -> str:
    value = equation(node)
    return value.split("->", 1)[1] if "->" in value else ""


def set_terms(node: onnx.NodeProto, new_terms: list[str]) -> None:
    old = equation(node)
    suffix = "->" + old.split("->", 1)[1] if "->" in old else ""
    value = ",".join(new_terms) + suffix
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
        np.issubdtype(array.dtype, np.floating)
        and array.ndim == 2
        and array.shape[0] == array.shape[1]
        and array.shape[0] > 1
        and np.all((array == 0) | (array == 1) | (array == -1))
        and np.all(np.count_nonzero(array, axis=0) == 1)
        and np.all(np.count_nonzero(array, axis=1) == 1)
    )


def transform_axis(
    array: np.ndarray,
    axis: int,
    permutation: np.ndarray,
    permutation_contracted_axis: int,
) -> np.ndarray:
    transformed = np.tensordot(
        array,
        permutation,
        axes=([axis], [permutation_contracted_axis]),
    )
    transformed = np.moveaxis(transformed, -1, axis)
    return transformed.astype(array.dtype, copy=False)


def make_plan(
    model: onnx.ModelProto,
    source_name: str,
    source_node_index: int,
    source_input_index: int,
    contracted_position: int,
    counts: Counter[str],
    use_locations: dict[str, list[tuple[int, int]]],
) -> dict[str, object] | None:
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    source = numpy_helper.to_array(initializers[source_name])
    node = graph.node[source_node_index]
    node_terms = terms(node)
    source_term = node_terms[source_input_index]
    if len(source_term) != 2 or source_term[0] == source_term[1]:
        return None
    contracted_label = source_term[contracted_position]
    replacement_label = source_term[1 - contracted_position]
    if contracted_label in output_term(node):
        return None
    target_occurrences: list[tuple[int, int]] = []
    for input_index, term in enumerate(node_terms):
        if input_index == source_input_index:
            continue
        target_occurrences.extend((input_index, position) for position, label in enumerate(term) if label == contracted_label)
    if len(target_occurrences) != 1:
        return None
    target_input_index, target_axis = target_occurrences[0]
    target_name = node.input[target_input_index]
    target_tensor = initializers.get(target_name)
    if target_tensor is None or counts[target_name] < 2:
        return None
    target = numpy_helper.to_array(target_tensor)
    target_term = node_terms[target_input_index]
    if (
        target.dtype != source.dtype
        or len(target_term) != target.ndim
        or replacement_label in target_term
        or target.shape[target_axis] != source.shape[contracted_position]
    ):
        return None
    transformed_target = transform_axis(target, target_axis, source, contracted_position)
    compensations: list[dict[str, object]] = []
    reserved = {source_name, target_name}
    for other_node_index, other_input_index in use_locations[target_name]:
        if other_node_index == source_node_index and other_input_index == target_input_index:
            continue
        other_node = graph.node[other_node_index]
        if other_node.op_type != "Einsum":
            return None
        other_terms = terms(other_node)
        if len(other_terms) != len(other_node.input):
            return None
        other_target_term = other_terms[other_input_index]
        if len(other_target_term) != target.ndim:
            return None
        axis_label = other_target_term[target_axis]
        if axis_label in output_term(other_node):
            return None
        options: list[tuple[int, int, int, str, np.ndarray]] = []
        for compensation_index, compensation_name in enumerate(other_node.input):
            if compensation_index == other_input_index or compensation_name in reserved:
                continue
            compensation_tensor = initializers.get(compensation_name)
            if compensation_tensor is None or counts[compensation_name] != 1:
                continue
            compensation = numpy_helper.to_array(compensation_tensor)
            compensation_term = other_terms[compensation_index]
            label_positions = [i for i, label in enumerate(compensation_term) if label == axis_label]
            if (
                compensation.dtype != source.dtype
                or len(compensation_term) != compensation.ndim
                or len(label_positions) != 1
            ):
                continue
            compensation_axis = label_positions[0]
            if compensation.shape[compensation_axis] != source.shape[contracted_position]:
                continue
            total_label_occurrences = sum(term.count(axis_label) for term in other_terms)
            if total_label_occurrences != 2:
                continue
            product = transform_axis(
                compensation,
                compensation_axis,
                source,
                contracted_position,
            )
            options.append(
                (
                    int(product.size),
                    compensation_index,
                    compensation_axis,
                    compensation_name,
                    product,
                )
            )
        if not options:
            return None
        _, compensation_index, compensation_axis, compensation_name, product = min(options, key=lambda item: item[0])
        reserved.add(compensation_name)
        compensations.append(
            {
                "node_index": other_node_index,
                "target_input_index": other_input_index,
                "target_axis": target_axis,
                "axis_label": axis_label,
                "compensation_input_index": compensation_index,
                "compensation_axis": compensation_axis,
                "compensation_name": compensation_name,
                "product": product,
            }
        )
    return {
        "source_name": source_name,
        "source_node_index": source_node_index,
        "source_input_index": source_input_index,
        "source_term": source_term,
        "contracted_position": contracted_position,
        "contracted_label": contracted_label,
        "replacement_label": replacement_label,
        "target_name": target_name,
        "target_input_index": target_input_index,
        "target_axis": target_axis,
        "target_term": target_term,
        "target_product": transformed_target,
        "compensations": compensations,
    }


def plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    counts, use_locations = locations(graph)
    result: list[dict[str, object]] = []
    for source_name, tensor in initializers.items():
        if counts[source_name] != 1:
            continue
        source = numpy_helper.to_array(tensor)
        if not is_signed_permutation(source):
            continue
        node_index, input_index = use_locations[source_name][0]
        if graph.node[node_index].op_type != "Einsum":
            continue
        for contracted_position in (0, 1):
            plan = make_plan(
                model,
                source_name,
                node_index,
                input_index,
                contracted_position,
                counts,
                use_locations,
            )
            if plan is not None:
                result.append(plan)
    return result


def build(model: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    graph = result.graph
    replacements = {
        str(plan["target_name"]): plan["target_product"],
        **{
            str(item["compensation_name"]): item["product"]
            for item in plan["compensations"]
        },
    }
    kept = []
    for tensor in graph.initializer:
        if tensor.name == plan["source_name"]:
            continue
        array = replacements.get(tensor.name)
        kept.append(numpy_helper.from_array(array, tensor.name) if isinstance(array, np.ndarray) else tensor)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    node = graph.node[int(plan["source_node_index"])]
    node_terms = terms(node)
    target_index = int(plan["target_input_index"])
    target_term = node_terms[target_index]
    axis = int(plan["target_axis"])
    node_terms[target_index] = target_term[:axis] + str(plan["replacement_label"]) + target_term[axis + 1 :]
    remove_index = int(plan["source_input_index"])
    new_inputs = [name for index, name in enumerate(node.input) if index != remove_index]
    new_terms = [term for index, term in enumerate(node_terms) if index != remove_index]
    del node.input[:]
    node.input.extend(new_inputs)
    set_terms(node, new_terms)
    onnx.checker.check_model(result, full_check=True)
    onnx.shape_inference.infer_shapes(result, strict_mode=True)
    return result


def concise(plan: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in plan.items()
        if key not in {"target_product", "compensations"}
    } | {
        "compensations": [
            {key: value for key, value in item.items() if key != "product"}
            for item in plan["compensations"]
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_costs = json.loads(args.base_costs.read_text())
    costs = raw_costs.get("costs") or {str(row["task"]): row["cost"] for row in raw_costs["ranked"]}
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.baseline) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            model = onnx.load_model_from_string(archive.read(member))
            for ordinal, plan in enumerate(plans(model), 1):
                try:
                    candidate = build(model, plan)
                    with tempfile.TemporaryDirectory(prefix=f"espa_{task:03d}_") as tmp:
                        probe = Path(tmp) / member
                        onnx.save(candidate, probe)
                        memory, params, cost = cost_of(str(probe))
                    baseline_cost = int(costs[str(task)])
                    if cost < 0 or cost >= baseline_cost:
                        continue
                    output = args.out_dir / f"task{task:03d}_r{ordinal:02d}.onnx"
                    onnx.save(candidate, output)
                    rows.append(
                        {
                            "task": task,
                            "path": str(output.relative_to(ROOT)),
                            "baseline_cost": baseline_cost,
                            "candidate_cost": int(cost),
                            "candidate_memory": int(memory),
                            "candidate_params": int(params),
                            "projected_gain": math.log(baseline_cost / cost),
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                            "rewrite": concise(plan),
                        }
                    )
                except Exception as exc:
                    errors.append({"task": task, "rewrite": concise(plan), "error": repr(exc)})
    payload = {
        "baseline": str(args.baseline),
        "candidate_count": len(rows),
        "projected_gain": sum(float(row["projected_gain"]) for row in rows),
        "rows": rows,
        "errors": errors,
    }
    manifest = args.out_dir / "build_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "projected_gain": payload["projected_gain"], "error_count": len(errors), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
