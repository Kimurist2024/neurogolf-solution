#!/usr/bin/env python3
"""Losslessly encode zero-heavy dense initializers as ONNX sparse initializers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "base_models"
TARGETS = {
    73: ("x",),
    260: ("E", "v1", "Flip", "gate"),
    271: ("zero3",),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_sparse(initializer: onnx.TensorProto) -> onnx.SparseTensorProto:
    array = numpy_helper.to_array(initializer)
    flat = array.reshape(-1)
    indices = np.flatnonzero(flat != 0).astype(np.int64)
    # The independent cost checker rejects a sparse value tensor with a zero
    # dimension.  All selected initializers have at least one nonzero value.
    values = flat[indices]
    value_tensor = numpy_helper.from_array(values, name=initializer.name)
    index_tensor = numpy_helper.from_array(indices, name=f"{initializer.name}_indices")
    return helper.make_sparse_tensor(value_tensor, index_tensor, list(array.shape))


def build(task: int, names: tuple[str, ...]) -> dict[str, object]:
    source = BASE / f"task{task:03d}.onnx"
    model = onnx.load(source)
    selected = {item.name: item for item in model.graph.initializer if item.name in names}
    if set(selected) != set(names):
        raise ValueError(f"task{task:03d}: missing initializers {set(names) - set(selected)}")

    conversions: list[dict[str, object]] = []
    retained = []
    for initializer in model.graph.initializer:
        if initializer.name not in selected:
            retained.append(initializer)
            continue
        array = numpy_helper.to_array(initializer)
        sparse = to_sparse(initializer)
        conversions.append(
            {
                "name": initializer.name,
                "shape": list(array.shape),
                "dense_elements": int(array.size),
                "sparse_values": int(np.count_nonzero(array)),
                "saved_params": int(array.size - np.count_nonzero(array)),
            }
        )
        model.graph.sparse_initializer.append(sparse)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    for item in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        for dim in item.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                raise ValueError(f"non-static shape: {item.name}")

    destination = HERE / "candidates" / f"task{task:03d}_sparse_exact.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return {
        "task": task,
        "source": str(source.relative_to(HERE.parents[3])),
        "source_sha256": sha256(source),
        "candidate": str(destination.relative_to(HERE.parents[3])),
        "candidate_sha256": sha256(destination),
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "conversions": conversions,
        "total_saved_params_static": sum(int(item["saved_params"]) for item in conversions),
    }


def main() -> None:
    rows = []
    for task, names in TARGETS.items():
        try:
            rows.append(build(task, names))
        except Exception as exc:  # noqa: BLE001 - rejection is part of the audit
            rows.append(
                {
                    "task": task,
                    "candidate": None,
                    "verdict": "REJECT",
                    "reason": "sparse initializer is not accepted as a dense tensor input",
                    "error": repr(exc),
                }
            )
    (HERE / "sparse_build.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
