#!/usr/bin/env python3
"""Extract immutable task366 authority member and inventory its structure."""

from __future__ import annotations

import collections
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ZIP = ROOT / "submission_base_8002.63.zip"
AUTH = HERE / "baseline_task366.onnx"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


with zipfile.ZipFile(ZIP) as archive:
    data = archive.read("task366.onnx")
AUTH.write_bytes(data)
model = onnx.load_from_string(data)
onnx.checker.check_model(model, full_check=True)
inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
by_value: dict[tuple[str, tuple[int, ...], bytes], list[str]] = collections.defaultdict(list)
for name, value in arrays.items():
    by_value[(str(value.dtype), tuple(value.shape), value.tobytes())].append(name)
duplicates = [names for names in by_value.values() if len(names) > 1]

used = collections.Counter(name for node in model.graph.node for name in node.input if name)
produced = {name for node in model.graph.node for name in node.output if name}
graph_outputs = {item.name for item in model.graph.output}
dead_nodes = [
    {"index": i, "op": node.op_type, "outputs": list(node.output)}
    for i, node in enumerate(model.graph.node)
    if all(out not in graph_outputs and used[out] == 0 for out in node.output if out)
]
unused_initializers = sorted(name for name in arrays if used[name] == 0)

type_shape = {}
for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
    tensor = value.type.tensor_type
    type_shape[value.name] = {
        "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
        "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in tensor.shape.dim],
    }

manifest = {
    "authority_zip": str(ZIP.relative_to(ROOT)),
    "authority_zip_sha256": sha(ZIP.read_bytes()),
    "member_path": str(AUTH.relative_to(ROOT)),
    "member_sha256": sha(data),
    "bytes": len(data),
    "ir_version": model.ir_version,
    "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
    "node_count": len(model.graph.node),
    "initializer_count": len(model.graph.initializer),
    "parameter_elements": int(sum(value.size for value in arrays.values())),
    "op_histogram": dict(collections.Counter(node.op_type for node in model.graph.node)),
    "initializer_shapes": {name: list(value.shape) for name, value in arrays.items()},
    "initializer_uses": dict(used),
    "exact_duplicate_initializer_groups": duplicates,
    "unused_initializers": unused_initializers,
    "dead_nodes": dead_nodes,
    "type_shape": type_shape,
}
(HERE / "authority_inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({key: manifest[key] for key in ("authority_zip_sha256", "member_sha256", "node_count", "initializer_count", "parameter_elements")}))
