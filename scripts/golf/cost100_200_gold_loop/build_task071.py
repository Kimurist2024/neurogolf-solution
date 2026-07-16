#!/usr/bin/env python3
"""Repair the truthful shapes on the exact task071 CastLike scalar shave.

The original scan removed the one-element ``i32zero`` initializer by replacing
``CastLike(x, i32zero)`` with ``Cast(x, to=INT32)``.  The arithmetic is the
same, but stale value_info shapes made ORT_DISABLE_ALL attempt an invalid
one-element buffer reuse.  This builder updates metadata only; the candidate's
nodes and initializers are otherwise unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/root_attr_constant_scan_155/candidates"
    / "task071_0027_CastLike.onnx"
)
OUTPUT = HERE / "candidates/task071_cost187_gold_shape_repaired.onnx"
EVIDENCE = HERE / "task071_build.json"


def set_shape(value: onnx.ValueInfoProto, dims: list[int]) -> None:
    shape = value.type.tensor_type.shape
    del shape.dim[:]
    for size in dims:
        shape.dim.add().dim_value = size


def main() -> None:
    model = onnx.load(SOURCE)
    values = {
        value.name: value
        for value in (
            list(model.graph.input)
            + list(model.graph.value_info)
            + list(model.graph.output)
        )
    }
    repaired = {
        "gather_u8_s": [30],
        "gather_i32": [30],
        "output": [1, 10, 30, 30],
    }
    for name, dims in repaired.items():
        set_shape(values[name], dims)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    EVIDENCE.write_text(
        json.dumps(
            {
                "source": str(SOURCE.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "output": str(OUTPUT.relative_to(ROOT)),
                "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                "metadata_repairs": repaired,
                "node_count": len(model.graph.node),
                "initializer_count": len(model.graph.initializer),
                "initializer_names": [item.name for item in model.graph.initializer],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
