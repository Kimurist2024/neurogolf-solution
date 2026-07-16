#!/usr/bin/env python3
"""Build the isolated task233 exact initializer-alias rewrite.

The two scalar initializers are byte-identical TensorProto values after their
names are erased. Repointing every use to one scalar and deleting the duplicate
therefore preserves the graph function for every possible input while reducing
the official parameter count by one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
OUT = HERE / "task233_exact_alias.onnx"


def tensor_key(tensor: onnx.TensorProto) -> bytes:
    clone = onnx.TensorProto()
    clone.CopyFrom(tensor)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def dims(value: onnx.ValueInfoProto) -> list[int]:
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise ValueError(f"non-positive/non-static shape: {value.name}")
        result.append(int(dim.dim_value))
    return result


def cost(model: onnx.ModelProto) -> dict[str, int]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    infos = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    excluded = {value.name for value in inferred.graph.input}
    excluded.update(value.name for value in inferred.graph.output)
    excluded.update(value.name for value in inferred.graph.initializer)
    memory = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in excluded or name in seen:
                continue
            seen.add(name)
            value = infos[name]
            dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            memory += math.prod(dims(value)) * np.dtype(dtype).itemsize
    params = sum(math.prod(t.dims) if t.dims else 1 for t in inferred.graph.initializer)
    return {"memory": memory, "params": params, "cost": memory + params}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        original_bytes = archive.read("task233.onnx")
    original = onnx.load_from_string(original_bytes)
    candidate = copy.deepcopy(original)

    initializers = {tensor.name: tensor for tensor in candidate.graph.initializer}
    canonical = initializers["one_i8"]
    duplicate = initializers["audit_one_i16"]
    if tensor_key(canonical) != tensor_key(duplicate):
        raise RuntimeError("the alias pair is no longer TensorProto-identical")
    if canonical.data_type != duplicate.data_type or list(canonical.dims) != list(duplicate.dims):
        raise RuntimeError("dtype/shape mismatch in alias pair")
    if not np.array_equal(numpy_helper.to_array(canonical), numpy_helper.to_array(duplicate)):
        raise RuntimeError("value mismatch in alias pair")

    replacements = 0
    for node in candidate.graph.node:
        for index, name in enumerate(node.input):
            if name == duplicate.name:
                node.input[index] = canonical.name
                replacements += 1
    if replacements == 0:
        raise RuntimeError("duplicate initializer had no consumer")
    kept = [tensor for tensor in candidate.graph.initializer if tensor.name != duplicate.name]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)

    onnx.checker.check_model(candidate, full_check=True)
    shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
    onnx.save(candidate, OUT)
    candidate_bytes = OUT.read_bytes()
    report = {
        "task": 233,
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": hashlib.sha256(BASE_ZIP.read_bytes()).hexdigest(),
        "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "candidate_path": str(OUT.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "rewrite": {
            "canonical": canonical.name,
            "removed_duplicate": duplicate.name,
            "tensorproto_identical_ignoring_name": True,
            "same_dtype": True,
            "same_shape": True,
            "same_value": True,
            "consumer_replacements": replacements,
        },
        "original_cost": cost(original),
        "candidate_cost": cost(candidate),
        "checker_full": True,
        "strict_data_prop": True,
    }
    (HERE / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
