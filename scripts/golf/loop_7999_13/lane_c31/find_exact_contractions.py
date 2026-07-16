#!/usr/bin/env python3
"""Enumerate exact constant-factor contractions that reduce stored parameters."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def inspect(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    node = model.graph.node[0]
    equation = next(attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation")
    left, output = equation.split("->")
    terms = left.split(",")
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    counts = Counter(index for term in terms for index in term)
    name_uses = Counter(node.input)
    dimensions: dict[str, int] = {}
    for name, term in zip(node.input, terms):
        if name not in arrays:
            continue
        for index, dimension in zip(term, arrays[name].shape):
            dimensions[index] = max(dimensions.get(index, 1), int(dimension))
    candidates = []
    for first, second in itertools.combinations(range(len(terms)), 2):
        name1, name2 = node.input[first], node.input[second]
        if name1 not in arrays or name2 not in arrays:
            continue
        term1, term2 = terms[first], terms[second]
        shared = set(term1) & set(term2)
        contract = sorted(index for index in shared if counts[index] == 2 and index not in output)
        if not contract:
            continue
        remaining = []
        for index in term1 + term2:
            if index not in contract and index not in remaining:
                remaining.append(index)
        new_size = math.prod(dimensions[index] for index in remaining)
        freed = 0
        if name_uses[name1] == 1:
            freed += arrays[name1].size
        if name2 != name1 and name_uses[name2] == 1:
            freed += arrays[name2].size
        candidates.append(
            {
                "operands": [first, second],
                "names": [name1, name2],
                "terms": [term1, term2],
                "contract": contract,
                "new_term": "".join(remaining),
                "new_size": int(new_size),
                "freed_unique_params": int(freed),
                "delta": int(new_size - freed),
            }
        )
    return {
        "path": str(path),
        "equation": equation,
        "occurrence_counts": counts,
        "pair_contractions": sorted(candidates, key=lambda row: row["delta"]),
    }


def main() -> None:
    result = {
        path.stem: inspect(path)
        for path in sorted((HERE / "baseline").glob("*.onnx"))
    }
    (HERE / "exact_contraction_inventory.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for task, row in result.items():
        print(task)
        for candidate in row["pair_contractions"][:20]:
            print(candidate)


if __name__ == "__main__":
    main()
