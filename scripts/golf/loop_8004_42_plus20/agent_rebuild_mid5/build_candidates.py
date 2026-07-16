#!/usr/bin/env python3
"""Build truthful controls and a generator-derived task367 rebuild."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent


def initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def clean_infer(model: onnx.ModelProto) -> onnx.ModelProto:
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)


def attach_truthful_baseline_types(model: onnx.ModelProto, task: int) -> onnx.ModelProto:
    """Attach measured runtime shapes for retained baseline-named tensors."""
    source = onnx.load(HERE / f"baseline_task{task:03d}.onnx")
    inferred_source = shape_inference.infer_shapes(
        source, strict_mode=False, data_prop=True
    )
    source_types = {
        value.name: value.type.tensor_type.elem_type
        for value in list(inferred_source.graph.value_info)
        + list(inferred_source.graph.output)
        if value.type.HasField("tensor_type")
    }
    runtime = json.loads((HERE / "baseline_runtime_trace.json").read_text())[
        str(task)
    ]["runtime"]
    del model.graph.value_info[:]
    for node in model.graph.node:
        for name in node.output:
            if name == "output":
                continue
            if name not in source_types or name not in runtime:
                continue
            model.graph.value_info.append(
                helper.make_tensor_value_info(
                    name, source_types[name], runtime[name]["shape"]
                )
            )
    onnx.checker.check_model(model, full_check=True)
    return shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)


def build_209() -> Path:
    model = onnx.load(HERE / "baseline_task209.onnx")
    remove_outputs = {
        "__cloak_sh_rbits",
        "__cloak_hid_rbits",
        "__cloak_sh_sraw",
        "__cloak_hid_sraw",
        "__cloak_hid_prawu0",
    }
    nodes = []
    for node in model.graph.node:
        if any(output in remove_outputs for output in node.output):
            continue
        item = copy.deepcopy(node)
        if item.output and item.output[0] == "rbits_bool":
            item.op_type = "Cast"
            del item.input[:]
            item.input.append("rbits")
            del item.attribute[:]
            item.attribute.append(helper.make_attribute("to", TensorProto.BOOL))
        elif item.output and item.output[0] == "sraw":
            item.op_type = "Cast"
            del item.input[:]
            item.input.append("srawf")
            del item.attribute[:]
            item.attribute.append(helper.make_attribute("to", TensorProto.UINT32))
        elif item.output and item.output[0] == "prawu":
            item.op_type = "Cast"
            del item.input[:]
            item.input.append("prawf")
            del item.attribute[:]
            item.attribute.append(helper.make_attribute("to", TensorProto.UINT32))
        nodes.append(item)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    )
    model = attach_truthful_baseline_types(model, 209)
    path = HERE / "candidate_task209_decloaked.onnx"
    onnx.save(model, path)
    return path


def build_187() -> Path:
    model = onnx.load(HERE / "baseline_task187.onnx")
    remove_outputs = {
        "seed_shape",
        "ish",
        "ihid",
        "rsh",
        "rhid",
        "mask_shape",
        "R4bh",
        "C4bh",
        "k4h",
    }
    nodes = []
    rewires = {
        "input16": ("input", "E"),
        "input32": ("input", "chmask_i32"),
        "rectF": ("rectB", "E"),
        "R4": ("R4b", "E"),
        "C4": ("C4b", "E"),
    }
    for node in model.graph.node:
        if any(output in remove_outputs for output in node.output):
            continue
        item = copy.deepcopy(node)
        if item.output and item.output[0] in rewires:
            del item.input[:]
            item.input.extend(rewires[item.output[0]])
        nodes.append(item)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.append(
        initializer("k4h", np.asarray([4], dtype=np.int64))
    )
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])
    )
    model = attach_truthful_baseline_types(model, 187)
    path = HERE / "candidate_task187_decloaked.onnx"
    onnx.save(model, path)
    return path


def build_367() -> Path:
    source = onnx.load(HERE / "baseline_task367.onnx")
    # Retain the proven row-mask rectangle scan, but replace every dynamic
    # CenterCropPad shape cloak with truthful bit-packing, Slice, and Pad.
    core = [copy.deepcopy(node) for node in source.graph.node[34:67]]
    for node in core:
        if node.output and node.output[0] == "lan64":
            node.input[1] = "sel"
    final = copy.deepcopy(source.graph.node[74])
    nodes = [
        helper.make_node(
            "Einsum",
            ["input", "packC", "packP"],
            ["BG30f"],
            equation="nchw,kc,w->nhk",
        ),
        helper.make_node(
            "Slice", ["BG30f", "s0", "s20", "ax1"], ["BG20f"]
        ),
        helper.make_node("Cast", ["BG20f"], ["BG20"], to=TensorProto.INT32),
        helper.make_node("Split", ["BG20"], ["B", "G"], axis=2, num_outputs=2),
        helper.make_node("Pad", ["G", "pad_top"], ["ab21"], mode="constant"),
        helper.make_node("Slice", ["ab21", "s0", "s20", "ax1"], ["ab"]),
        helper.make_node("Slice", ["G", "s1", "s20", "ax1"], ["be19"]),
        helper.make_node("Pad", ["be19", "pad_bottom"], ["be"], mode="constant"),
        *core,
        final,
    ]
    wanted = {"w", "pr", "one8", "bfalse", "two_i32", "axd"}
    old_init = {
        item.name: copy.deepcopy(item)
        for item in source.graph.initializer
        if item.name in wanted
    }
    selector6 = onnx.numpy_helper.to_array(
        next(item for item in source.graph.initializer if item.name == "selc")
    )
    selector = np.pad(selector6, ((0, 0), (0, 4), (0, 0), (0, 0)))
    pack_c = np.zeros((2, 10), dtype=np.float32)
    pack_c[0, 0] = 1.0
    pack_c[1, 5] = 1.0
    pack_p = np.zeros(30, dtype=np.float32)
    pack_p[:20] = np.asarray([1 << index for index in range(20)], dtype=np.float32)
    initializers = [
        *old_init.values(),
        initializer("packC", pack_c),
        initializer("packP", pack_p),
        initializer("s0", np.asarray([0], dtype=np.int64)),
        initializer("s1", np.asarray([1], dtype=np.int64)),
        initializer("s20", np.asarray([20], dtype=np.int64)),
        initializer("ax1", np.asarray([1], dtype=np.int64)),
        initializer("pad_top", np.asarray([0, 1, 0, 0, 0, 0], dtype=np.int64)),
        initializer("pad_bottom", np.asarray([0, 0, 0, 0, 1, 0], dtype=np.int64)),
        initializer("sel", selector.astype(np.uint64)),
    ]
    graph = helper.make_graph(
        nodes,
        "task367_truthful_rowmask",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT64, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 23)],
        ir_version=11,
    )
    model = clean_infer(model)
    path = HERE / "candidate_task367_truthful_rowmask.onnx"
    onnx.save(model, path)
    return path


def static_cost(path: Path) -> dict[str, int]:
    model = onnx.load(path)
    params = sum(int(np.prod(item.dims)) for item in model.graph.initializer)
    initializers = {item.name for item in model.graph.initializer}
    memory = 0
    for value in model.graph.value_info:
        if value.name in initializers:
            continue
        tensor = value.type.tensor_type
        shape = [int(dim.dim_value) for dim in tensor.shape.dim]
        itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type)).itemsize
        memory += int(np.prod(shape)) * itemsize
    return {"params": params, "memory": memory, "cost": params + memory}


def main() -> None:
    report = {}
    for task, builder in ((187, build_187), (209, build_209), (367, build_367)):
        try:
            path = builder()
            report[str(task)] = {"path": str(path), **static_cost(path)}
        except Exception as exc:  # noqa: BLE001
            report[str(task)] = {"error": f"{type(exc).__name__}: {exc}"}
        print(task, report[str(task)])
    (HERE / "candidate_build_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
