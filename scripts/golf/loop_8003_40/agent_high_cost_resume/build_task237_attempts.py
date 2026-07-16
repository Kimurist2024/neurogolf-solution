#!/usr/bin/env python3
"""Build generator-domain exact task237 guard-removal attempts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8003_40/base_models/task237.onnx"
OUT = HERE / "attempts"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_elements(tensor: onnx.TensorProto) -> int:
    return math.prod(tensor.dims) if tensor.dims else 1


def static_cost(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    infos = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    free = {value.name for value in inferred.graph.input} | {
        value.name for value in inferred.graph.output
    }
    initializers = {value.name for value in inferred.graph.initializer}
    memory = 0
    seen = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in seen or name in free or name in initializers:
                continue
            seen.add(name)
            tensor_type = infos[name].type.tensor_type
            elements = math.prod(int(dim.dim_value) for dim in tensor_type.shape.dim)
            width = np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
            ).itemsize
            memory += elements * width
    params = sum(tensor_elements(value) for value in inferred.graph.initializer)
    return {"memory": memory, "params": params, "cost": memory + params}


def conv_bias(model: onnx.ModelProto) -> list[list[object]]:
    spec = importlib.util.spec_from_file_location(
        "check_conv_bias", ROOT / "scripts/golf/check_conv_bias.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [list(row) for row in module.check_model(model)]


def replace_all_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def remove_min(model: onnx.ModelProto) -> None:
    nodes = list(model.graph.node)
    target = next(node for node in nodes if node.op_type == "Min" and node.output[0] == "start_for_ray")
    # Generator invariant: every marker column is sampled in [0,width-2].
    # Therefore its packed start code never exceeds the active last column.
    replace_all_uses(model, target.output[0], target.input[0])
    del model.graph.node[:]
    model.graph.node.extend(node for node in nodes if node is not target)


def remove_shrink_with_shift(model: onnx.ModelProto) -> None:
    nodes = list(model.graph.node)
    target = next(
        node for node in nodes if node.op_type == "Shrink" and node.output[0] == "max_col_index_base"
    )
    # Shrink(lambda=0,bias=15) is max_q-15 because a valid grid always has
    # max_q>0.  Move +15 to the constant comparison indices instead.
    replace_all_uses(model, target.output[0], target.input[0])
    initializer = next(value for value in model.graph.initializer if value.name == "col_index_u8")
    shifted = numpy_helper.to_array(initializer).copy() + np.uint8(15)
    initializer.CopyFrom(numpy_helper.from_array(shifted, initializer.name))
    del model.graph.node[:]
    model.graph.node.extend(node for node in nodes if node is not target)


def build(label: str, transforms: list) -> dict[str, object]:
    baseline = onnx.load(SOURCE, load_external_data=False)
    candidate = copy.deepcopy(baseline)
    for transform in transforms:
        transform(candidate)
    onnx.checker.check_model(candidate, full_check=True)
    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
    bias = conv_bias(candidate)
    path = OUT / f"task237_{label}.onnx"
    onnx.save(candidate, path)
    before = static_cost(baseline)
    after = static_cost(candidate)
    return {
        "task": 237,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "transforms": [transform.__name__ for transform in transforms],
        "baseline": before,
        "candidate": after,
        "cost_reduction": before["cost"] - after["cost"],
        "checker_full": "PASS",
        "strict_shape_inference_data_prop": "PASS",
        "conv_bias_ub": bias,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        build("remove_min", [remove_min]),
        build("shift_shrink", [remove_shrink_with_shift]),
        build("combined", [remove_min, remove_shrink_with_shift]),
    ]
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha(SOURCE),
        "attempts": rows,
    }
    (HERE / "task237_build_attempts.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
