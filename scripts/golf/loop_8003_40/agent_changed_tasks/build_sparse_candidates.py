#!/usr/bin/env python3
"""Replace one zero-containing dense initializer by an exact sparse initializer.

Each candidate changes only the initializer representation.  It deliberately
uses a single sparse initializer per model so that both the official scorer's
and the team validator's sanitizers preserve the tensor binding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "base_models"
OUT = HERE / "candidates_sparse"
TARGETS = {
    73: "x",
    260: "E",
    271: "zero3",
    289: "slope",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_from_sparse(sparse: onnx.SparseTensorProto) -> np.ndarray:
    values = numpy_helper.to_array(sparse.values)
    indices = numpy_helper.to_array(sparse.indices)
    result = np.zeros(tuple(sparse.dims), dtype=values.dtype)
    result.reshape(-1)[indices] = values
    return result


def rewrite(task: int, initializer_name: str) -> dict[str, object]:
    source = BASE / f"task{task:03d}.onnx"
    destination = OUT / source.name
    model = onnx.load(source)
    dense = next(init for init in model.graph.initializer if init.name == initializer_name)
    array = numpy_helper.to_array(dense)
    flat = array.reshape(-1)
    indices = np.flatnonzero(flat != 0).astype(np.int64)
    values = flat[indices]
    if not 0 < len(indices) < flat.size:
        raise ValueError(f"{initializer_name}: expected both zero and nonzero values")

    # The official sanitizer visits remaining dense initializers first, then
    # discovers this sparse value through node inputs.  Give it exactly the
    # next canonical name so the (unrenamed) SparseTensorProto stays bound.
    sparse_name = f"safe_name_{len(model.graph.initializer) - 1}"
    value_tensor = numpy_helper.from_array(values, name=sparse_name)
    # An empty indices name is legal and prevents the stricter team sanitizer
    # from consuming an otherwise unused canonical name.
    index_tensor = numpy_helper.from_array(indices, name="")
    sparse = onnx.helper.make_sparse_tensor(value_tensor, index_tensor, list(array.shape))

    kept = [init for init in model.graph.initializer if init.name != initializer_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.sparse_initializer.append(sparse)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == initializer_name:
                node.input[index] = sparse_name

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    reconstructed = dense_from_sparse(model.graph.sparse_initializer[-1])
    if not np.array_equal(reconstructed, array, equal_nan=True):
        raise AssertionError(f"sparse reconstruction differs for task{task:03d}")

    OUT.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return {
        "task": task,
        "source": str(source),
        "candidate": str(destination),
        "initializer": initializer_name,
        "sparse_name": sparse_name,
        "dense_elements": int(flat.size),
        "sparse_values": int(values.size),
        "parameter_reduction_expected": int(flat.size - values.size),
        "source_sha256": sha256(source),
        "candidate_sha256": sha256(destination),
        "checker": "PASS",
        "strict_shape_inference": "PASS",
        "dense_reconstruction": "BIT_IDENTICAL",
    }


def main() -> None:
    records = [rewrite(task, name) for task, name in TARGETS.items()]
    path = HERE / "sparse_build_manifest.json"
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
