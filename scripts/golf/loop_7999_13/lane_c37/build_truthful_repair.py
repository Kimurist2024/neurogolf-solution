#!/usr/bin/env python3
"""Repair task162 CSE models without relying on CenterCropPad shape tricks.

The source models feed a length-one target tensor to CenterCropPad calls with
one, two, or three axes and declare almost every runtime tensor as 1x1x1x1.
ORT_DISABLE_ALL broadcasts this unofficially, while ORT_DEFAULT rejects the
contract.  This builder replaces each target with an exact-length constant,
removes the now-dead ConstantOfShape nodes, discards stale value_info, and
stores strict inferred shapes for every intermediate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.rank_dir import cost_of  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair(source: Path, output: Path) -> dict[str, object]:
    model = onnx.load(source)
    target_values: dict[str, int] = {}
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if initializer.name == "sh":
            target_values[initializer.name] = int(array.reshape(-1)[0])
    for node in model.graph.node:
        if node.op_type != "ConstantOfShape":
            continue
        value = next(attr for attr in node.attribute if attr.name == "value")
        target_values[node.output[0]] = int(numpy_helper.to_array(value.t).reshape(-1)[0])

    names = {initializer.name for initializer in model.graph.initializer}
    replacements: dict[tuple[str, int], str] = {}
    added: list[onnx.TensorProto] = []
    for node in model.graph.node:
        if node.op_type != "CenterCropPad":
            continue
        axes = next(tuple(attr.ints) for attr in node.attribute if attr.name == "axes")
        old = node.input[1]
        if old not in target_values:
            raise ValueError(f"unknown target tensor {old!r}")
        key = (old, len(axes))
        name = replacements.setdefault(key, f"{old}_truth_len{len(axes)}")
        node.input[1] = name
        if name not in names:
            added.append(
                numpy_helper.from_array(
                    np.full(len(axes), target_values[old], dtype=np.int64), name=name
                )
            )
            names.add(name)

    kept_nodes = [node for node in model.graph.node if node.op_type != "ConstantOfShape"]
    model.graph.ClearField("node")
    model.graph.node.extend(kept_nodes)
    used = {name for node in model.graph.node for name in node.input}
    kept_initializers = [item for item in model.graph.initializer if item.name in used]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept_initializers + added)

    model.graph.ClearField("value_info")
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    model.graph.value_info.extend(inferred.graph.value_info)
    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    memory, params, total = cost_of(output)
    return {
        "source": str(source.resolve().relative_to(ROOT)),
        "output": str(output.resolve().relative_to(ROOT)),
        "source_sha256": digest(source),
        "output_sha256": digest(output),
        "nodes_before": len(onnx.load(source).graph.node),
        "nodes_after": len(model.graph.node),
        "constant_of_shape_removed": sum(
            node.op_type == "ConstantOfShape" for node in onnx.load(source).graph.node
        ),
        "target_replacements": [
            {"source": old, "axis_count": count, "initializer": name}
            for (old, count), name in sorted(replacements.items())
        ],
        "value_info_count": len(model.graph.value_info),
        "all_value_info_static_positive": all(
            dim.HasField("dim_value") and dim.dim_value > 0
            for value in model.graph.value_info
            for dim in value.type.tensor_type.shape.dim
        ),
        "cost": {"memory": memory, "params": params, "total": total},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = repair(args.source, args.output)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
