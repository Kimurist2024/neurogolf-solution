#!/usr/bin/env python3
"""Freeze task196 authority and inventory the only historical sub-1210 lead."""

from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path
import zipfile

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8002.63.zip"
HISTORICAL = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task196_r07_static296.onnx"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attr_key(node: onnx.NodeProto) -> bytes:
    clone = onnx.NodeProto()
    clone.CopyFrom(node)
    del clone.input[:]
    del clone.output[:]
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def inventory(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    types = {}
    for value in [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]:
        if value.type.HasField("tensor_type"):
            types[value.name] = value.type.tensor_type.elem_type
    init_names = {init.name for init in model.graph.initializer}
    uses = collections.Counter(
        name for node in model.graph.node for name in node.input if name
    )
    consumers = collections.Counter(
        name for node in model.graph.node for name in node.input if name
    )
    consumers.update(output.name for output in model.graph.output)
    init_groups: dict[str, list[str]] = collections.defaultdict(list)
    for init in model.graph.initializer:
        init_groups[hashlib.sha256(init.SerializeToString(deterministic=True)).hexdigest()].append(init.name)
    exact_seen: dict[tuple[object, ...], str] = {}
    exact_cse: list[dict[str, str]] = []
    for node in model.graph.node:
        if len(node.output) != 1 or not node.output[0]:
            continue
        key = (node.domain, node.op_type, tuple(node.input), attr_key(node))
        if key in exact_seen:
            exact_cse.append({"output": node.output[0], "reuse": exact_seen[key]})
        else:
            exact_seen[key] = node.output[0]
    same_casts = []
    for node in model.graph.node:
        if node.op_type == "Cast" and len(node.input) == len(node.output) == 1:
            target = next((attr.i for attr in node.attribute if attr.name == "to"), None)
            if target == types.get(node.input[0]):
                same_casts.append(node.output[0])
    same_castlikes = [
        node.output[0]
        for node in model.graph.node
        if node.op_type == "CastLike"
        and len(node.input) == 2
        and len(node.output) == 1
        and types.get(node.input[0]) == types.get(node.input[1])
    ]
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params": sum(math.prod(init.dims) for init in model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "ops": dict(collections.Counter(node.op_type for node in model.graph.node)),
        "identities": [node.output[0] for node in model.graph.node if node.op_type == "Identity"],
        "same_type_casts": same_casts,
        "same_type_castlikes": same_castlikes,
        "exact_cse": exact_cse,
        "duplicate_initializer_groups": [group for group in init_groups.values() if len(group) > 1],
        "unused_initializers": sorted(init_names - uses.keys()),
        "dead_node_outputs": sorted(
            output
            for node in model.graph.node
            for output in node.output
            if output and consumers[output] == 0
        ),
        "initializer_shapes": {
            init.name: {
                "dtype": onnx.TensorProto.DataType.Name(init.data_type),
                "shape": list(init.dims),
                "elements": math.prod(init.dims),
                "uses": uses[init.name],
            }
            for init in model.graph.initializer
        },
    }


def main() -> None:
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        payload = archive.read("task196.onnx")
    authority = HERE / "baseline_task196.onnx"
    authority.write_bytes(payload)
    report = {
        "authority_zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
        "authority_zip_sha256": sha(AUTHORITY_ZIP),
        "authority": inventory(authority),
        "historical_sub1210": inventory(HISTORICAL),
    }
    (HERE / "inventory.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
