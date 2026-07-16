#!/usr/bin/env python3
"""Reproducible exact-rank and static-precontraction audit for B7 baselines."""

from __future__ import annotations

import collections
import itertools
import json
import math
from pathlib import Path

import numpy as np
import onnx
import sympy
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
TASKS = (30, 132, 175, 199, 212, 240, 304)
GENERATOR = {
    30: "1caeab9d",
    132: "56ff96f3",
    175: "73251a56",
    199: "834ec97d",
    212: "8d510a79",
    240: "9d9215db",
    304: "c3e719e8",
}


def exact_rank(array: np.ndarray, axis: int) -> int:
    matrix = np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1)
    exact = sympy.Matrix(
        [[sympy.Rational(float(value)) for value in row] for row in matrix]
    )
    return int(exact.rank())


def precontraction_savings(
    model: onnx.ModelProto, arrays: dict[str, np.ndarray], terms: list[str], rhs: str
) -> list[dict[str, object]]:
    names = list(arrays)
    dimensions: dict[str, int] = {}
    for name, raw_term in zip(model.graph.node[0].input, terms, strict=True):
        term = raw_term.replace("...", "")
        shape = (1, 10, 30, 30) if name == "input" else arrays[name].shape
        shape = shape[-len(term) :] if term else ()
        for label, dimension in zip(term, shape, strict=True):
            dimensions[label] = max(dimensions.get(label, 1), int(dimension))

    normalized = [term.replace("...", "") for term in terms]
    all_counts = collections.Counter(label for term in normalized for label in term)
    results: list[dict[str, object]] = []
    for count in range(1, len(names) + 1):
        for selected_tuple in itertools.combinations(names, count):
            selected = set(selected_tuple)
            indices = [
                index for index, name in enumerate(model.graph.node[0].input) if name in selected
            ]
            parent = {index: index for index in indices}

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(left: int, right: int) -> None:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parent[right_root] = left_root

            for offset, left in enumerate(indices):
                for right in indices[offset + 1 :]:
                    if set(normalized[left]) & set(normalized[right]):
                        union(left, right)
            components: dict[int, list[int]] = collections.defaultdict(list)
            for index in indices:
                components[find(index)].append(index)

            replacement_cost = 0
            for component in components.values():
                component_counts = collections.Counter(
                    label for index in component for label in normalized[index]
                )
                boundary = {
                    label
                    for label in component_counts
                    if all_counts[label] > component_counts[label] or label in rhs
                }
                replacement_cost += math.prod(dimensions[label] for label in boundary)
            original_cost = sum(int(arrays[name].size) for name in selected)
            saving = original_cost - replacement_cost
            if saving > 0:
                results.append(
                    {
                        "initializers": sorted(selected),
                        "original_cost": original_cost,
                        "replacement_cost": replacement_cost,
                        "saving": saving,
                    }
                )
    return sorted(results, key=lambda item: int(item["saving"]), reverse=True)


def main() -> None:
    report: list[dict[str, object]] = []
    for task in TASKS:
        model = onnx.load(HERE / f"baseline_task{task:03d}.onnx")
        node = model.graph.node[0]
        equation = next(
            onnx.helper.get_attribute_value(attribute).decode()
            for attribute in node.attribute
            if attribute.name == "equation"
        )
        lhs, rhs = equation.split("->")
        terms = lhs.split(",")
        arrays = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in model.graph.initializer
        }
        initializers: list[dict[str, object]] = []
        for name, array in arrays.items():
            mode_ranks = [exact_rank(array, axis) for axis in range(array.ndim)]
            old_cost = int(array.size)
            factor_costs = [
                int(array.shape[axis] * rank + rank * (old_cost // array.shape[axis]))
                for axis, rank in enumerate(mode_ranks)
            ]
            initializers.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "size": old_cost,
                    "exact_mode_ranks": mode_ranks,
                    "best_single_mode_factor_cost": min(factor_costs),
                    "best_single_mode_saving": old_cost - min(factor_costs),
                }
            )
        same_shape_pairs = []
        for left, right in itertools.combinations(arrays, 2):
            if arrays[left].shape == arrays[right].shape:
                same_shape_pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "shape": list(arrays[left].shape),
                        "identical": bool(np.array_equal(arrays[left], arrays[right])),
                    }
                )
        contractions = precontraction_savings(model, arrays, terms, rhs)
        report.append(
            {
                "task": task,
                "generator_hash": GENERATOR[task],
                "initializers": initializers,
                "same_shape_pairs": same_shape_pairs,
                "positive_exact_static_precontractions": contractions,
            }
        )
    output = HERE / "factor_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
