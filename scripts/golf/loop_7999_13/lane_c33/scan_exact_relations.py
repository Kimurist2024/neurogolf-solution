#!/usr/bin/env python3
"""Run the C32 exact alias/scalar/diagonal relation scanner on C33."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "lane_c32" / "scan_exact_relations.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("c33_rel_shared", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError(SHARED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {}
    for path in sorted((HERE / "baseline").glob("*.onnx")):
        model = onnx.load(path)
        arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
        aliases = []
        diagonals = []
        repeated_rows = []
        for first, second in itertools.combinations(arrays, 2):
            left, right = arrays[first], arrays[second]
            if left.ndim == right.ndim:
                for permutation in itertools.permutations(range(right.ndim)):
                    moved = np.transpose(right, permutation)
                    if moved.shape != left.shape:
                        continue
                    ratio = module.proportional(left, moved)
                    if ratio is not None:
                        aliases.append(
                            {
                                "left": first,
                                "right": second,
                                "permutation": list(permutation),
                                "ratio": ratio,
                                "signed": ratio in (-1.0, 1.0),
                                "power2": ratio != 0 and np.log2(abs(ratio)).is_integer(),
                            }
                        )
            relation = module.diagonal_equivalence(left, right)
            if relation is not None:
                diagonals.append({"left": first, "right": second, **relation})
            if left.ndim == right.ndim == 2 and left.shape[1] == right.shape[1]:
                for i, j in itertools.product(range(left.shape[0]), range(right.shape[0])):
                    if np.array_equal(left[i], right[j]):
                        repeated_rows.append(
                            {"left": first, "left_row": i, "right": second, "right_row": j}
                        )
        result[path.stem] = {
            "exact_or_scalar_aliases": aliases,
            "diagonal_equivalences": diagonals,
            "cross_initializer_repeated_rows": repeated_rows,
        }
    (HERE / "exact_relations.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
