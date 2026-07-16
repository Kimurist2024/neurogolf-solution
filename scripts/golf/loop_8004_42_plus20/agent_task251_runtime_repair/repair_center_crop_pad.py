#!/usr/bin/env python3
"""Repair task251 archive r03 CenterCropPad shape-vector arity.

This produces two deliberately distinct artifacts:

* ``task251_r03_arity_repaired_cloaked.onnx`` keeps the archive's false
  value_info and exists only to isolate the default-ORT shape-arity failure.
* ``task251_r03_arity_repaired_truthful.onnx`` discards the false value_info
  and regenerates every intermediate type/shape with strict data propagation.

Only the second artifact is eligible for the runtime-shape safety gate.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = (
    ROOT
    / "scripts/golf/loop_7999_13/lane_archive_loose_sweep"
    / "task251_r03_static406.onnx"
)
CLOAKED = HERE / "task251_r03_arity_repaired_cloaked.onnx"
TRUTHFUL = HERE / "task251_r03_arity_repaired_truthful.onnx"


def axes_of(node: onnx.NodeProto) -> list[int]:
    for attr in node.attribute:
        if attr.name == "axes":
            return [int(value) for value in attr.ints]
    return []


def constant_of_shape_values(model: onnx.ModelProto) -> dict[str, int]:
    values: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type != "ConstantOfShape" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            array = numpy_helper.to_array(attr.t)
            if array.size == 1:
                values[node.output[0]] = int(array.reshape(-1)[0])
    return values


def repair_arity(model: onnx.ModelProto) -> list[dict[str, object]]:
    scalar_values = constant_of_shape_values(model)
    initializers = {item.name for item in model.graph.initializer}
    added: dict[tuple[str, int], str] = {}
    repairs: list[dict[str, object]] = []

    for index, node in enumerate(model.graph.node):
        if node.op_type != "CenterCropPad":
            continue
        axes = axes_of(node)
        shape_name = node.input[1]
        if len(axes) <= 1 or shape_name not in scalar_values:
            continue
        key = (shape_name, len(axes))
        vector_name = added.get(key)
        if vector_name is None:
            vector_name = f"{shape_name}_arity{len(axes)}"
            if vector_name in initializers:
                raise RuntimeError(f"initializer collision: {vector_name}")
            value = scalar_values[shape_name]
            model.graph.initializer.append(
                numpy_helper.from_array(
                    np.full((len(axes),), value, dtype=np.int64),
                    name=vector_name,
                )
            )
            initializers.add(vector_name)
            added[key] = vector_name
        node.input[1] = vector_name
        repairs.append(
            {
                "node_index": index,
                "node_name": node.name,
                "axes": axes,
                "old_shape_input": shape_name,
                "new_shape_input": vector_name,
                "target": [scalar_values[shape_name]] * len(axes),
            }
        )
    return repairs


def tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=True
    )
    return {
        value.name: int(value.type.tensor_type.elem_type)
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
        if value.type.HasField("tensor_type")
    }


def trace_shapes(model: onnx.ModelProto) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Trace every node output after removing all stale shape declarations."""

    types = tensor_types(model)
    traced = copy.deepcopy(model)
    del traced.graph.value_info[:]
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if not name or name in names:
                continue
            if name not in types:
                raise RuntimeError(f"missing inferred tensor type for {name}")
            traced.graph.output.append(
                onnx.helper.make_tensor_value_info(name, types[name], None)
            )
            names.append(name)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    arrays = session.run(
        names,
        {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)},
    )
    return (
        {name: list(np.asarray(array).shape) for name, array in zip(names, arrays)},
        types,
    )


def make_truthful(model: onnx.ModelProto) -> onnx.ModelProto:
    shapes, types = trace_shapes(model)
    truthful = copy.deepcopy(model)
    del truthful.graph.value_info[:]
    output_names = {value.name for value in truthful.graph.output}
    for node in truthful.graph.node:
        for name in node.output:
            if not name or name in output_names:
                continue
            truthful.graph.value_info.append(
                onnx.helper.make_tensor_value_info(name, types[name], shapes[name])
            )
    onnx.checker.check_model(truthful, full_check=True)
    reinferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(truthful), strict_mode=True, data_prop=True
    )
    for value in list(reinferred.graph.value_info) + list(reinferred.graph.output):
        if not value.type.tensor_type.HasField("shape"):
            raise RuntimeError(f"strict inference lost shape for {value.name}")
        if any(
            not dim.HasField("dim_value") or dim.dim_value <= 0
            for dim in value.type.tensor_type.shape.dim
        ):
            raise RuntimeError(f"non-static shape for {value.name}")
    return truthful


def main() -> None:
    source = onnx.load(SOURCE)
    repaired = copy.deepcopy(source)
    repairs = repair_arity(repaired)
    if len(repairs) != 52:
        raise RuntimeError(f"expected 52 arity repairs, found {len(repairs)}")
    # The arity-only diagnostic intentionally retains the archive's stale
    # [1,1,1,1] value_info, so a full checker correctly rejects it after the
    # shape vectors become statically visible.  Save it only to isolate ORT
    # behavior; it is never an eligible candidate.
    onnx.save(repaired, CLOAKED)

    truthful = make_truthful(repaired)
    onnx.save(truthful, TRUTHFUL)

    print(f"source={SOURCE.relative_to(ROOT)}")
    print(f"repairs={len(repairs)}")
    for key, name in sorted(
        ((key, name) for key, name in {
            (item["old_shape_input"], len(item["axes"])): item["new_shape_input"]
            for item in repairs
        }.items()),
        key=lambda item: (str(item[0][0]), int(item[0][1])),
    ):
        print(f"shape {key[0]} arity={key[1]} -> {name}")
    print(f"cloaked={CLOAKED.relative_to(ROOT)}")
    print(f"truthful={TRUTHFUL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
