#!/usr/bin/env python3
"""Build exact sparse-initializer rewrites for the three structurally safe nets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
TARGETS = {40: "H", 176: "E", 252: "W"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_from_sparse(item: onnx.SparseTensorProto) -> np.ndarray:
    values = numpy_helper.to_array(item.values)
    indices = numpy_helper.to_array(item.indices)
    dense = np.zeros(tuple(item.dims), dtype=values.dtype)
    dense.reshape(-1)[indices] = values
    return dense


def rewrite(task: int, name: str) -> dict[str, object]:
    source = HERE / "base" / f"task{task:03d}.onnx"
    destination = HERE / "candidates" / f"task{task:03d}_sparse_{name}.onnx"
    model = onnx.load(source)
    original = next(item for item in model.graph.initializer if item.name == name)
    array = numpy_helper.to_array(original)
    flat = array.reshape(-1)
    indices = np.flatnonzero(flat != 0).astype(np.int64)
    values = flat[indices]
    if not 0 < values.size < flat.size:
        raise RuntimeError(f"{name} is not sparse")

    kept = [item for item in model.graph.initializer if item.name != name]
    # scripts.lib.scoring.sanitize_model visits the kept dense initializers
    # first.  Naming the sparse values with the next canonical identifier
    # preserves its binding even though that sanitizer does not rename sparse
    # initializer names directly.
    sparse_name = f"safe_name_{len(kept)}"
    sparse = onnx.helper.make_sparse_tensor(
        numpy_helper.from_array(values, name=sparse_name),
        numpy_helper.from_array(indices, name=""),
        list(array.shape),
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.sparse_initializer.append(sparse)
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == name:
                node.input[index] = sparse_name

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    if not np.array_equal(array, dense_from_sparse(model.graph.sparse_initializer[-1])):
        raise AssertionError("sparse reconstruction differs")
    onnx.save(model, destination)
    return {
        "task": task,
        "source": str(source),
        "candidate": str(destination),
        "initializer": name,
        "dense_elements": int(flat.size),
        "sparse_values": int(values.size),
        "expected_parameter_reduction": int(flat.size - values.size),
        "source_sha256": sha(source),
        "candidate_sha256": sha(destination),
        "checker_full": True,
        "strict_data_prop": True,
        "dense_reconstruction": "BIT_IDENTICAL",
    }


def main() -> None:
    rows = []
    for task, name in TARGETS.items():
        try:
            rows.append(rewrite(task, name))
        except Exception as exc:
            rows.append(
                {
                    "task": task,
                    "initializer": name,
                    "candidate": None,
                    "decision": "REJECT",
                    "error": f"{type(exc).__name__}: {exc}",
                    "reason": "full checker/strict inference cannot infer an Einsum operand rank from SparseTensorProto",
                }
            )
    (HERE / "sparse_build_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
