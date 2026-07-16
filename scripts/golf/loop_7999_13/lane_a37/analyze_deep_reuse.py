#!/usr/bin/env python3
"""Audit exact initializer views and Einsum uses for the A37 safe candidates."""

from __future__ import annotations

import itertools
import json
import string
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    13: ROOT / "lane_initializer_contraction_wave17" / "task013_combined.onnx",
    105: ROOT / "lane_initializer_contraction_wave17" / "task105_combined.onnx",
}
LABELS = string.ascii_letters


def equation(node: onnx.NodeProto) -> str:
    return next(attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation")


def assignments(target: np.ndarray, source: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
    """Return source-axis assignments whose contraction is target times a signed 2^k."""
    if target.dtype != source.dtype or source.ndim > 6 or target.ndim > source.ndim:
        return []
    target_labels = LABELS[: target.ndim]
    reduction_labels = LABELS[target.ndim : target.ndim + source.ndim]
    candidates: list[tuple[int, ...]] = []

    def walk(axis: int, current: list[int], reduction_dims: list[int]) -> None:
        if axis == source.ndim:
            if all(item in current for item in range(target.ndim)):
                candidates.append(tuple(current))
            return
        dim = source.shape[axis]
        for target_axis, target_dim in enumerate(target.shape):
            if dim == target_dim:
                current.append(target_axis)
                walk(axis + 1, current, reduction_dims)
                current.pop()
        for group, reduction_dim in enumerate(reduction_dims):
            if dim == reduction_dim:
                current.append(-1 - group)
                walk(axis + 1, current, reduction_dims)
                current.pop()
        current.append(-1 - len(reduction_dims))
        reduction_dims.append(dim)
        walk(axis + 1, current, reduction_dims)
        reduction_dims.pop()
        current.pop()

    walk(0, [], [])
    result: list[tuple[tuple[int, ...], float]] = []
    for assignment in candidates:
        source_subscript = "".join(
            target_labels[item] if item >= 0 else reduction_labels[-1 - item]
            for item in assignment
        )
        try:
            view = np.asarray(
                np.einsum(f"{source_subscript}->{target_labels}", source, optimize=False),
                dtype=target.dtype,
            )
        except (TypeError, ValueError):
            continue
        if view.shape != target.shape:
            continue
        ratios: list[float] = []
        valid = True
        for target_value, source_value in zip(target.reshape(-1), view.reshape(-1)):
            tv = float(target_value)
            sv = float(source_value)
            if tv == 0.0 and sv == 0.0:
                continue
            if tv == 0.0 or sv == 0.0:
                valid = False
                break
            ratios.append(tv / sv)
        if not valid or not ratios:
            continue
        scale = ratios[0]
        if not all(value == scale for value in ratios):
            continue
        magnitude = abs(scale)
        if magnitude == 0.0 or not np.isfinite(magnitude):
            continue
        exponent = np.log2(magnitude)
        if exponent != round(exponent):
            continue
        rebuilt = np.asarray(view * scale, dtype=target.dtype)
        if np.array_equal(rebuilt, target, equal_nan=True):
            result.append((assignment, scale))
    return sorted(set(result))


def audit(task: int, path: Path) -> dict[str, object]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes: dict[str, list[int | str]] = {}
    for value in itertools.chain(
        inferred.graph.input,
        inferred.graph.output,
        inferred.graph.value_info,
    ):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shapes[value.name] = [
            int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
            for dim in tensor_type.shape.dim
        ]
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    uses: dict[str, list[dict[str, object]]] = {name: [] for name in arrays}
    nodes: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        row: dict[str, object] = {
            "index": index,
            "op": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        if node.op_type == "Einsum":
            row["equation"] = equation(node)
            terms = equation(node).split("->", 1)[0].split(",")
            label_uses: dict[str, list[dict[str, object]]] = {}
            for input_index, (name, term) in enumerate(zip(node.input, terms)):
                shape = list(arrays[name].shape) if name in arrays else shapes.get(name, [])
                for axis, label in enumerate(term):
                    label_uses.setdefault(label, []).append(
                        {
                            "input": input_index,
                            "name": name,
                            "axis": axis,
                            "dim": shape[axis] if axis < len(shape) else "?",
                            "initializer": name in arrays,
                        }
                    )
            row["labels"] = label_uses
            local_collapses: list[dict[str, object]] = []
            rhs = equation(node).split("->", 1)[1]
            global_counts = {
                label: sum(term.count(label) for term in terms)
                for label in set("".join(terms))
            }
            for input_index, (name, term) in enumerate(zip(node.input, terms)):
                if name not in arrays or len(set(term)) != len(term):
                    continue
                array = arrays[name]
                for axis, private_label in enumerate(term):
                    if private_label in rhs or global_counts[private_label] != 1:
                        continue
                    kept = term[:axis] + term[axis + 1 :]
                    try:
                        original = np.einsum(f"{term}->{kept}", array, optimize=False)
                    except (TypeError, ValueError):
                        continue
                    for tied_axis, tied_label in enumerate(term):
                        if tied_axis == axis or array.shape[tied_axis] != array.shape[axis]:
                            continue
                        replacement = term[:axis] + tied_label + term[axis + 1 :]
                        try:
                            collapsed = np.einsum(f"{replacement}->{kept}", array, optimize=False)
                        except (TypeError, ValueError):
                            continue
                        if np.array_equal(original, collapsed, equal_nan=True):
                            local_collapses.append(
                                {
                                    "input": input_index,
                                    "name": name,
                                    "term": term,
                                    "private_label": private_label,
                                    "replacement_label": tied_label,
                                    "replacement_term": replacement,
                                }
                            )
            row["local_collapses"] = local_collapses
            pair_collapses: list[dict[str, object]] = []
            private_axes: list[tuple[int, str, str, int]] = []
            for input_index, (name, term) in enumerate(zip(node.input, terms)):
                if name not in arrays or len(set(term)) != len(term):
                    continue
                for axis, label in enumerate(term):
                    if label not in rhs and global_counts[label] == 1:
                        private_axes.append((input_index, name, term, axis))
            for left, right in itertools.combinations(private_axes, 2):
                left_index, left_name, left_term, left_axis = left
                right_index, right_name, right_term, right_axis = right
                left_array = arrays[left_name]
                right_array = arrays[right_name]
                if left_array.shape[left_axis] != right_array.shape[right_axis]:
                    continue
                left_private = left_term[left_axis]
                right_private = right_term[right_axis]
                all_kept = "".join(dict.fromkeys(
                    left_term.replace(left_private, "") + right_term.replace(right_private, "")
                ))
                try:
                    original = np.einsum(
                        f"{left_term},{right_term}->{all_kept}",
                        left_array,
                        right_array,
                        optimize=False,
                    )
                    tied_term = right_term[:right_axis] + left_private + right_term[right_axis + 1 :]
                    collapsed = np.einsum(
                        f"{left_term},{tied_term}->{all_kept}",
                        left_array,
                        right_array,
                        optimize=False,
                    )
                except (TypeError, ValueError):
                    continue
                if np.array_equal(original, collapsed, equal_nan=True):
                    pair_collapses.append(
                        {
                            "left_input": left_index,
                            "left_name": left_name,
                            "left_term": left_term,
                            "kept_label": left_private,
                            "right_input": right_index,
                            "right_name": right_name,
                            "right_term": right_term,
                            "freed_label": right_private,
                            "right_replacement_term": tied_term,
                        }
                    )
            row["pair_collapses"] = pair_collapses
        nodes.append(row)
        for input_index, name in enumerate(node.input):
            if name in uses:
                entry: dict[str, object] = {"node": index, "input": input_index, "op": node.op_type}
                if node.op_type == "Einsum":
                    entry["term"] = equation(node).split("->", 1)[0].split(",")[input_index]
                    entry["equation"] = equation(node)
                uses[name].append(entry)
    relations: list[dict[str, object]] = []
    for target_name, target in arrays.items():
        for source_name, source in arrays.items():
            if source_name == target_name:
                continue
            for assignment, scale in assignments(target, source):
                relations.append(
                    {
                        "target": target_name,
                        "target_shape": list(target.shape),
                        "target_size": int(target.size),
                        "source": source_name,
                        "source_shape": list(source.shape),
                        "assignment": list(assignment),
                        "scale": scale,
                    }
                )
    return {
        "task": task,
        "path": str(path),
        "initializers": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size": int(array.size),
                "values": array.tolist(),
                "uses": uses[name],
            }
            for name, array in arrays.items()
        },
        "relations": relations,
        "nodes": nodes,
    }


def main() -> None:
    output = {f"task{task:03d}": audit(task, path) for task, path in MODELS.items()}
    destination = Path(__file__).with_name("deep_reuse_audit.json")
    destination.write_text(json.dumps(output, indent=2) + "\n")
    summary = {
        key: {
            "initializers": len(value["initializers"]),
            "relations": value["relations"],
        }
        for key, value in output.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
