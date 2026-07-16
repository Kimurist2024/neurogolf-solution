#!/usr/bin/env python3
"""Enumerate exact per-node constant contractions with global-use cost accounting."""

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
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    global_uses = Counter(name for node in model.graph.node for name in node.input)
    rows = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        equation = next(attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation")
        left, output = equation.split("->")
        terms = left.split(",")
        counts = Counter(label for term in terms for label in term)
        dimensions = {}
        for name, term in zip(node.input, terms):
            if name not in arrays:
                continue
            for label, dimension in zip(term, arrays[name].shape):
                dimensions[label] = max(dimensions.get(label, 1), int(dimension))
        for first, second in itertools.combinations(range(len(terms)), 2):
            name1, name2 = node.input[first], node.input[second]
            if name1 not in arrays or name2 not in arrays:
                continue
            shared = set(terms[first]) & set(terms[second])
            contract = sorted(label for label in shared if counts[label] == 2 and label not in output)
            if not contract:
                continue
            remaining = []
            for label in terms[first] + terms[second]:
                if label not in contract and label not in remaining:
                    remaining.append(label)
            new_size = math.prod(dimensions[label] for label in remaining)
            freed = 0
            if global_uses[name1] == 1:
                freed += arrays[name1].size
            if name2 != name1 and global_uses[name2] == 1:
                freed += arrays[name2].size
            rows.append(
                {
                    "node": node_index,
                    "operands": [first, second],
                    "names": [name1, name2],
                    "terms": [terms[first], terms[second]],
                    "contract": contract,
                    "new_term": "".join(remaining),
                    "new_size": int(new_size),
                    "freed_global_params": int(freed),
                    "delta": int(new_size - freed),
                }
            )
    return {"pairs": sorted(rows, key=lambda row: row["delta"])}


def main() -> None:
    result = {path.stem: inspect(path) for path in sorted((HERE / "baseline").glob("*.onnx"))}
    (HERE / "exact_contractions.json").write_text(json.dumps(result, indent=2) + "\n")
    for task, row in result.items():
        print(task)
        for item in row["pairs"][:20]:
            print(item)


if __name__ == "__main__":
    main()
