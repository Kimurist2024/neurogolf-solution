#!/usr/bin/env python3
"""Freeze task125 authority and inventory archive/sound candidates."""

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
ARCHIVE_DIR = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200"
MODELS = {
    **{f"archive_r{index:02d}": ARCHIVE_DIR / f"task125_r{index:02d}_static{static}.onnx"
       for index, static in enumerate((162, 167, 169, 170, 171, 171, 175, 186), 1)},
    "sound_pool14": ROOT / "scripts/golf/scratch_codex/task125/task125_pool14.onnx",
}


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
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }
    uses = collections.Counter(name for node in model.graph.node for name in node.input if name)
    consumers = collections.Counter(name for node in model.graph.node for name in node.input if name)
    consumers.update(value.name for value in model.graph.output)
    init_names = {init.name for init in model.graph.initializer}
    groups: dict[bytes, list[str]] = collections.defaultdict(list)
    for init in model.graph.initializer:
        clone = onnx.TensorProto()
        clone.CopyFrom(init)
        clone.name = ""
        groups[clone.SerializeToString(deterministic=True)].append(init.name)
    seen: dict[tuple[object, ...], str] = {}
    cse = []
    for node in model.graph.node:
        if len(node.output) != 1 or not node.output[0]:
            continue
        key = (node.domain, node.op_type, tuple(node.input), attr_key(node))
        if key in seen:
            cse.append({"output": node.output[0], "reuse": seen[key]})
        else:
            seen[key] = node.output[0]
    same_casts = []
    same_castlikes = []
    for node in model.graph.node:
        if node.op_type == "Cast" and len(node.input) == len(node.output) == 1:
            target = next((attr.i for attr in node.attribute if attr.name == "to"), None)
            if target == types.get(node.input[0]):
                same_casts.append(node.output[0])
        if (
            node.op_type == "CastLike"
            and len(node.input) == 2
            and len(node.output) == 1
            and types.get(node.input[0]) == types.get(node.input[1])
        ):
            same_castlikes.append(node.output[0])
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
        "exact_cse": cse,
        "duplicate_initializer_groups": [names for names in groups.values() if len(names) > 1],
        "unused_initializers": sorted(init_names - uses.keys()),
        "dead_outputs": sorted(
            output for node in model.graph.node for output in node.output if output and consumers[output] == 0
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
        payload = archive.read("task125.onnx")
    authority = HERE / "baseline_task125.onnx"
    authority.write_bytes(payload)
    rows = {"authority": inventory(authority)}
    rows.update({label: inventory(path) for label, path in MODELS.items()})
    report = {
        "authority_zip": str(AUTHORITY_ZIP.relative_to(ROOT)),
        "authority_zip_sha256": sha(AUTHORITY_ZIP),
        "models": rows,
    }
    (HERE / "inventory.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
