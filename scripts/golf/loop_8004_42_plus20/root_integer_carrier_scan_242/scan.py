#!/usr/bin/env python3
"""Prove and narrow closed integer carrier components in all authority models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


SIGNED_LIMITS = {
    onnx.TensorProto.INT8: (-128, 127),
    onnx.TensorProto.INT16: (-32768, 32767),
    onnx.TensorProto.INT32: (-(1 << 31), (1 << 31) - 1),
    onnx.TensorProto.INT64: (-(1 << 63), (1 << 63) - 1),
}
TARGETS = {
    onnx.TensorProto.INT64: (
        onnx.TensorProto.INT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
    ),
    onnx.TensorProto.INT32: (onnx.TensorProto.INT8, onnx.TensorProto.INT16),
}

# Slots where int32 and int64 are alternative, semantically identical index
# representations.  This is deliberately narrower than the ONNX schemas: a
# Cast may be removed only when every use is an index/shape parameter, never a
# data operand.
FLEXIBLE_INDEX_SLOTS = {
    "CenterCropPad": {1},
    "Gather": {1},
    "GatherElements": {1},
    "GatherND": {1},
    "ScatterElements": {1},
    "Slice": {1, 2, 3, 4},
}

# These operations propagate the dtype of their data input(s) to output 0.
DATA0_OUTPUT = {
    "Abs", "CenterCropPad", "Compress", "CumSum", "Expand", "Flatten",
    "Gather", "GatherElements", "GatherND", "Identity", "Neg", "Pad",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
    "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare",
    "Reshape", "ReverseSequence", "Shrink", "Sign", "Slice", "Squeeze",
    "Tile", "Transpose", "Trilu", "Unsqueeze",
}
HOMOGENEOUS = {
    "Add", "BitShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "Div", "Max",
    "Mean", "Min", "Mod", "Mul", "PRelu", "Pow", "Sub", "Sum",
}
PRESERVE_INTERVAL = DATA0_OUTPUT - {
    "CumSum", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
    "ReduceMean", "ReduceProd", "ReduceSum", "ReduceSumSquare",
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        if value:
            self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, values: Iterable[str]) -> None:
        present = [value for value in values if value in self.parent]
        if not present:
            return
        root = self.find(present[0])
        for value in present[1:]:
            other = self.find(value)
            if other != root:
                self.parent[other] = root


@dataclass
class Component:
    source_dtype: int
    names: set[str]
    node_outputs: list[tuple[int, str]]
    initializers: list[str]
    interval: tuple[int, int]
    roots: list[dict[str, Any]]
    output_bytes: int


def dims(value: onnx.ValueInfoProto) -> list[int] | None:
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def elements(value: onnx.ValueInfoProto) -> int | None:
    shape = dims(value)
    return math.prod(shape) if shape is not None else None


def tensor_type_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    return {
        value.name: value
        for value in list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    }


def dtype_name(dtype: int) -> str:
    return onnx.TensorProto.DataType.Name(dtype)


def tensor_attr(node: onnx.NodeProto, attr_name: str = "value") -> onnx.TensorProto | None:
    for attr in node.attribute:
        if attr.name == attr_name and attr.type == onnx.AttributeProto.TENSOR:
            return attr.t
    return None


def producer_map(model: onnx.ModelProto) -> dict[str, tuple[int, onnx.NodeProto, int]]:
    return {
        name: (node_index, node, output_index)
        for node_index, node in enumerate(model.graph.node)
        for output_index, name in enumerate(node.output)
        if name
    }


def data_groups(node: onnx.NodeProto) -> list[list[str]]:
    values = list(node.input)
    outputs = list(node.output)
    if not outputs:
        return []
    if node.op_type in HOMOGENEOUS:
        return [[*values, outputs[0]]]
    if node.op_type in DATA0_OUTPUT and values:
        return [[values[0], outputs[0]]]
    if node.op_type == "Where" and len(values) >= 3:
        return [[values[1], values[2], outputs[0]]]
    if node.op_type == "Concat":
        return [[*values, outputs[0]]]
    if node.op_type == "CastLike" and len(values) >= 2:
        return [[values[1], outputs[0]]]
    if node.op_type == "ScatterElements" and len(values) >= 3:
        return [[values[0], values[2], outputs[0]]]
    if node.op_type == "ScatterND" and len(values) >= 3:
        return [[values[0], values[2], outputs[0]]]
    if node.op_type == "TopK" and values:
        return [[values[0], outputs[0]]]
    if node.op_type == "MaxPool" and values:
        return [[values[0], outputs[0]]]
    if node.op_type == "Split" and values:
        return [[values[0], *outputs]]
    return []


def source_dtype_range(dtype: int) -> tuple[int, int] | None:
    ranges = {
        onnx.TensorProto.BOOL: (0, 1),
        onnx.TensorProto.INT8: (-128, 127),
        onnx.TensorProto.UINT8: (0, 255),
        onnx.TensorProto.INT16: (-32768, 32767),
        onnx.TensorProto.UINT16: (0, 65535),
        onnx.TensorProto.INT32: (-(1 << 31), (1 << 31) - 1),
    }
    return ranges.get(dtype)


def exact_array_interval(array: np.ndarray) -> tuple[int, int] | None:
    if array.dtype.kind not in "iub" or array.size == 0:
        return None
    return int(array.min()), int(array.max())


def constant_node_array(node: onnx.NodeProto) -> np.ndarray | None:
    value = tensor_attr(node)
    if value is not None:
        return np.asarray(numpy_helper.to_array(value))
    for attr in node.attribute:
        if attr.name == "value_int":
            return np.asarray(attr.i, dtype=np.int64)
        if attr.name == "value_ints":
            return np.asarray(attr.ints, dtype=np.int64)
    return None


def component_interval(
    model: onnx.ModelProto,
    typed: dict[str, onnx.ValueInfoProto],
    component: set[str],
    source_dtype: int,
) -> tuple[tuple[int, int] | None, list[dict[str, Any]]]:
    intervals: dict[str, tuple[int, int]] = {}
    roots: list[dict[str, Any]] = []
    initializers = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    producers = producer_map(model)

    for name in component:
        if name in initializers:
            interval = exact_array_interval(initializers[name])
            if interval is None:
                return None, roots
            intervals[name] = interval
            roots.append({"kind": "initializer", "name": name, "interval": list(interval)})

    for node_index, node in enumerate(model.graph.node):
        owned = [name for name in node.output if name in component]
        if not owned:
            continue
        op = node.op_type
        interval: tuple[int, int] | None = None
        if op == "Cast":
            source = typed.get(node.input[0])
            source_dtype_value = source.type.tensor_type.elem_type if source is not None else None
            interval = source_dtype_range(source_dtype_value or 0)
            if interval is not None:
                roots.append(
                    {
                        "kind": "cast",
                        "node": node_index,
                        "source_dtype": dtype_name(source_dtype_value),
                        "interval": list(interval),
                    }
                )
        elif op == "Constant":
            array = constant_node_array(node)
            interval = exact_array_interval(array) if array is not None else None
            if interval is not None:
                roots.append({"kind": "constant", "node": node_index, "interval": list(interval)})
        elif op == "ConstantOfShape":
            value = tensor_attr(node)
            array = np.asarray(numpy_helper.to_array(value)) if value is not None else None
            interval = exact_array_interval(array) if array is not None else None
            if interval is not None:
                roots.append({"kind": "constant_of_shape", "node": node_index, "interval": list(interval)})
        elif op in {"Add", "Sub", "Mul"}:
            inputs = [intervals.get(name) for name in node.input]
            if inputs and all(value is not None for value in inputs):
                low, high = inputs[0]  # type: ignore[misc]
                for right_low, right_high in inputs[1:]:  # type: ignore[misc]
                    if op == "Add":
                        low, high = low + right_low, high + right_high
                    elif op == "Sub":
                        low, high = low - right_high, high - right_low
                    else:
                        products = (low * right_low, low * right_high, high * right_low, high * right_high)
                        low, high = min(products), max(products)
                interval = (low, high)
        elif op in {"Min", "Max", "Sum", "Mean", "Where", "Concat", "ScatterElements", "ScatterND", "Split"}:
            relevant = [intervals[name] for name in node.input if name in component and name in intervals]
            if relevant:
                interval = (min(value[0] for value in relevant), max(value[1] for value in relevant))
        elif op in PRESERVE_INTERVAL or op in {"CastLike", "TopK", "MaxPool"}:
            relevant = [intervals[name] for name in node.input if name in component and name in intervals]
            if relevant:
                interval = (min(value[0] for value in relevant), max(value[1] for value in relevant))
        elif op == "ReduceSum":
            source_interval = intervals.get(node.input[0]) if node.input else None
            source_info = typed.get(node.input[0]) if node.input else None
            output_info = typed.get(node.output[0]) if node.output else None
            source_elements = elements(source_info) if source_info is not None else None
            output_elements = elements(output_info) if output_info is not None else None
            if source_interval and source_elements and output_elements and source_elements % output_elements == 0:
                count = source_elements // output_elements
                interval = (source_interval[0] * count, source_interval[1] * count)
        if interval is None:
            return None, roots
        for name in owned:
            intervals[name] = interval

    if any(name not in intervals for name in component if name in producers or name in initializers):
        return None, roots
    values = [intervals[name] for name in component if name in intervals]
    if not values:
        return None, roots
    return (min(value[0] for value in values), max(value[1] for value in values)), roots


def components_for_dtype(
    model: onnx.ModelProto,
    inferred: onnx.ModelProto,
    source_dtype: int,
) -> list[Component]:
    typed = tensor_type_map(inferred)
    names = {
        name for name, value in typed.items()
        if value.type.tensor_type.elem_type == source_dtype
    }
    names.update(
        item.name for item in model.graph.initializer if item.data_type == source_dtype
    )
    union = UnionFind()
    for name in names:
        union.add(name)
    for node in model.graph.node:
        for group in data_groups(node):
            union.union(group)
    grouped: dict[str, set[str]] = defaultdict(set)
    for name in names:
        grouped[union.find(name)].add(name)

    graph_inputs = {value.name for value in model.graph.input}
    graph_outputs = {value.name for value in model.graph.output}
    initializers = {item.name for item in model.graph.initializer}
    producers = producer_map(model)
    allowed_producers = DATA0_OUTPUT | HOMOGENEOUS | {
        "Cast", "CastLike", "Concat", "Constant", "ConstantOfShape", "MaxPool",
        "ScatterElements", "ScatterND", "Split", "TopK", "Where",
    }
    result: list[Component] = []
    for component in grouped.values():
        if component & graph_inputs or component & graph_outputs:
            continue
        fixed = False
        node_outputs: list[tuple[int, str]] = []
        for name in component:
            producer = producers.get(name)
            if producer is None:
                if name not in initializers:
                    fixed = True
                continue
            node_index, node, output_index = producer
            node_outputs.append((node_index, name))
            if node.op_type not in allowed_producers:
                fixed = True
            if node.op_type in {"TopK", "MaxPool"} and output_index != 0:
                fixed = True
            if node.op_type == "Cast":
                source = typed.get(node.input[0])
                source_type = source.type.tensor_type.elem_type if source is not None else 0
                if source_dtype_range(source_type) is None:
                    fixed = True
        if fixed:
            continue
        interval, roots = component_interval(model, typed, component, source_dtype)
        if interval is None:
            continue
        # A component made only of graph initializers still has a real parameter
        # and memory-cost opportunity.  Count both stored tensors and produced
        # intermediates so such components are not silently dropped.
        output_bytes = sum(
            int(np.asarray(numpy_helper.to_array(item)).nbytes)
            for item in model.graph.initializer
            if item.name in component
        )
        for _, name in node_outputs:
            value = typed.get(name)
            count = elements(value) if value is not None else None
            if count is None:
                fixed = True
                break
            output_bytes += count * np.dtype(helper.tensor_dtype_to_np_dtype(source_dtype)).itemsize
        if fixed or output_bytes == 0:
            continue
        result.append(
            Component(
                source_dtype=source_dtype,
                names=component,
                node_outputs=node_outputs,
                initializers=sorted(component & initializers),
                interval=interval,
                roots=roots,
                output_bytes=output_bytes,
            )
        )
    return result


def replace_tensor_dtype(tensor: onnx.TensorProto, target: int) -> onnx.TensorProto:
    array = np.asarray(numpy_helper.to_array(tensor))
    target_dtype = np.dtype(helper.tensor_dtype_to_np_dtype(target))
    converted = array.astype(target_dtype)
    if not np.array_equal(converted.astype(array.dtype), array):
        raise ValueError(f"initializer {tensor.name} is not exactly representable as {dtype_name(target)}")
    return numpy_helper.from_array(converted, tensor.name)


def set_constant_dtype(node: onnx.NodeProto, target: int) -> None:
    value = tensor_attr(node)
    if value is not None:
        replacement = replace_tensor_dtype(value, target)
        for attr in node.attribute:
            if attr.name == "value":
                attr.t.CopyFrom(replacement)
                return
    array = constant_node_array(node)
    if array is None:
        raise ValueError("unsupported Constant representation")
    target_dtype = np.dtype(helper.tensor_dtype_to_np_dtype(target))
    converted = array.astype(target_dtype)
    del node.attribute[:]
    node.attribute.append(helper.make_attribute("value", numpy_helper.from_array(converted)))


def build_component(model: onnx.ModelProto, component: Component, target: int) -> onnx.ModelProto:
    low, high = SIGNED_LIMITS[target]
    if not (low <= component.interval[0] and component.interval[1] <= high):
        raise ValueError("proved interval does not fit target")
    candidate = copy.deepcopy(model)
    names = component.names
    kept: list[onnx.TensorProto] = []
    for initializer in candidate.graph.initializer:
        kept.append(replace_tensor_dtype(initializer, target) if initializer.name in names else initializer)
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)
    for node in candidate.graph.node:
        if not any(name in names for name in node.output):
            continue
        if node.op_type == "Cast":
            attr = next(item for item in node.attribute if item.name == "to")
            attr.i = target
        elif node.op_type == "Constant":
            set_constant_dtype(node, target)
        elif node.op_type == "ConstantOfShape":
            value = tensor_attr(node)
            if value is None:
                raise ValueError("ConstantOfShape lacks typed value")
            replacement = replace_tensor_dtype(value, target)
            next(attr for attr in node.attribute if attr.name == "value").t.CopyFrom(replacement)
    for value in candidate.graph.value_info:
        if value.name in names:
            value.type.tensor_type.elem_type = target
    return candidate


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"carrier242_{task:03d}_{label}_") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def session_smoke(model: onnx.ModelProto) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def static_arg_index_interval(
    node: onnx.NodeProto,
    typed: dict[str, onnx.ValueInfoProto],
) -> tuple[int, int] | None:
    """Return the exact range bound for a statically shaped ArgMin/ArgMax."""
    if node.op_type not in {"ArgMin", "ArgMax"} or not node.input:
        return None
    source = typed.get(node.input[0])
    shape = dims(source) if source is not None else None
    if not shape:
        return None
    axis = next((int(attr.i) for attr in node.attribute if attr.name == "axis"), 0)
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        return None
    return (0, shape[axis] - 1)


def removable_index_casts(
    model: onnx.ModelProto,
    inferred: onnx.ModelProto,
) -> list[dict[str, Any]]:
    """Find int64->int32 Casts removable without changing index semantics."""
    typed = tensor_type_map(inferred)
    producers = producer_map(model)
    consumers: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            consumers[name].append((node_index, node.op_type, input_index))
    graph_outputs = {value.name for value in model.graph.output}
    result: list[dict[str, Any]] = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Cast" or not node.input or not node.output:
            continue
        target = next((int(attr.i) for attr in node.attribute if attr.name == "to"), 0)
        source = typed.get(node.input[0])
        source_dtype = source.type.tensor_type.elem_type if source is not None else 0
        output = node.output[0]
        uses = consumers.get(output, [])
        if (
            source_dtype != onnx.TensorProto.INT64
            or target != onnx.TensorProto.INT32
            or output in graph_outputs
            or not uses
            or not all(slot in FLEXIBLE_INDEX_SLOTS.get(op, set()) for _, op, slot in uses)
        ):
            continue
        producer = producers.get(node.input[0])
        interval = static_arg_index_interval(producer[1], typed) if producer is not None else None
        if interval is None or interval[1] > SIGNED_LIMITS[onnx.TensorProto.INT32][1]:
            continue
        result.append(
            {
                "node_index": node_index,
                "source": node.input[0],
                "output": output,
                "interval": list(interval),
                "producer": producer[1].op_type,
                "uses": [
                    {"node_index": use_node, "op_type": op, "input_index": slot}
                    for use_node, op, slot in uses
                ],
            }
        )
    return result


def build_index_cast_removal(model: onnx.ModelProto, record: dict[str, Any]) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    node_index = int(record["node_index"])
    source = str(record["source"])
    output = str(record["output"])
    nodes: list[onnx.NodeProto] = []
    for current_index, node in enumerate(candidate.graph.node):
        if current_index == node_index:
            continue
        for input_index, name in enumerate(node.input):
            if name == output:
                node.input[input_index] = source
        nodes.append(node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    values = [value for value in candidate.graph.value_info if value.name != output]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(values)
    return candidate


def cast_census(model: onnx.ModelProto, inferred: onnx.ModelProto) -> dict[str, int]:
    typed = tensor_type_map(inferred)
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    result = Counter()
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        result["cast_nodes"] += 1
        source = typed.get(node.input[0])
        source_dtype = source.type.tensor_type.elem_type if source is not None else 0
        target = next((attr.i for attr in node.attribute if attr.name == "to"), 0)
        if source_dtype == target:
            result["same_dtype_casts"] += 1
        downstream = consumers.get(node.output[0], [])
        if downstream and all(item.op_type in {"Cast", "CastLike"} for item in downstream):
            result["cast_only_consumers"] += 1
    return dict(result)


def component_record(component: Component) -> dict[str, Any]:
    return {
        "source_dtype": dtype_name(component.source_dtype),
        "name_count": len(component.names),
        "node_output_count": len(component.node_outputs),
        "initializer_count": len(component.initializers),
        "interval": list(component.interval),
        "source_output_bytes": component.output_bytes,
        "roots": component.roots,
        "names": sorted(component.names),
    }


def main() -> int:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    census = Counter()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    strict_failures: list[dict[str, Any]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        census["authority_models"] = len(members)
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            source_bytes = archive.read(member)
            model = onnx.load_model_from_string(source_bytes)
            try:
                inferred = onnx.shape_inference.infer_shapes(
                    copy.deepcopy(model), strict_mode=True, data_prop=True
                )
            except Exception as exc:  # noqa: BLE001
                strict_failures.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
                continue
            census.update(cast_census(model, inferred))
            double_initializers = sum(item.data_type == onnx.TensorProto.DOUBLE for item in model.graph.initializer)
            if double_initializers:
                census["models_with_double_initializer"] += 1
                census["double_initializers"] += double_initializers
            task_components: list[Component] = []
            for source_dtype in TARGETS:
                found = components_for_dtype(model, inferred, source_dtype)
                census[f"proved_{dtype_name(source_dtype).lower()}_components"] += len(found)
                task_components.extend(found)
            cast_removals = removable_index_casts(model, inferred)
            census["proved_redundant_index_casts"] += len(cast_removals)
            if not task_components and not cast_removals:
                continue
            baseline = profile(model, task, "base")
            best: tuple[int, onnx.ModelProto, dict[str, Any]] | None = None
            for ordinal, component in enumerate(task_components, 1):
                base_record = component_record(component)
                for target in TARGETS[component.source_dtype]:
                    target_low, target_high = SIGNED_LIMITS[target]
                    if component.interval[0] < target_low or component.interval[1] > target_high:
                        census["range_rejections"] += 1
                        continue
                    try:
                        candidate = build_component(model, component, target)
                        onnx.checker.check_model(candidate, full_check=True)
                        onnx.shape_inference.infer_shapes(
                            candidate, strict_mode=True, data_prop=True
                        )
                        session_smoke(candidate)
                        current = profile(candidate, task, f"c{ordinal}_{dtype_name(target)}")
                        row = {
                            "task": task,
                            "authority_member_sha256": hashlib.sha256(source_bytes).hexdigest(),
                            "component_ordinal": ordinal,
                            **base_record,
                            "target_dtype": dtype_name(target),
                            "baseline": baseline,
                            "candidate": current,
                            "memory_nonincrease": current["memory"] <= baseline["memory"],
                            "strict_lower": current["cost"] < baseline["cost"],
                            "checker_full": True,
                            "strict_shape_inference_data_prop": True,
                            "ort_disable_all_session": True,
                        }
                        rows.append(row)
                        if row["strict_lower"] and row["memory_nonincrease"]:
                            saving = baseline["cost"] - current["cost"]
                            if best is None or saving > best[0]:
                                best = (saving, candidate, row)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(
                            {
                                "task": task,
                                "component_ordinal": ordinal,
                                "source_dtype": dtype_name(component.source_dtype),
                                "target_dtype": dtype_name(target),
                                "interval": list(component.interval),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
            for ordinal, removal in enumerate(cast_removals, 1):
                try:
                    candidate = build_index_cast_removal(model, removal)
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(
                        candidate, strict_mode=True, data_prop=True
                    )
                    session_smoke(candidate)
                    current = profile(candidate, task, f"remove_cast_{ordinal}")
                    row = {
                        "task": task,
                        "authority_member_sha256": hashlib.sha256(source_bytes).hexdigest(),
                        "method": "remove_redundant_index_cast",
                        "removal_ordinal": ordinal,
                        "source_dtype": "INT64",
                        "target_dtype": "INT32",
                        **removal,
                        "baseline": baseline,
                        "candidate": current,
                        "memory_nonincrease": current["memory"] <= baseline["memory"],
                        "strict_lower": current["cost"] < baseline["cost"],
                        "checker_full": True,
                        "strict_shape_inference_data_prop": True,
                        "ort_disable_all_session": True,
                    }
                    rows.append(row)
                    if row["strict_lower"] and row["memory_nonincrease"]:
                        saving = baseline["cost"] - current["cost"]
                        if best is None or saving > best[0]:
                            best = (saving, candidate, row)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "task": task,
                            "method": "remove_redundant_index_cast",
                            "removal_ordinal": ordinal,
                            "source_dtype": "INT64",
                            "target_dtype": "INT32",
                            "interval": removal["interval"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if best is not None:
                saving, candidate, row = best
                path = CANDIDATES / f"task{task:03d}_integer_carrier.onnx"
                onnx.save(candidate, path)
                row["path"] = str(path.relative_to(ROOT))
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                row["projected_gain"] = math.log(row["baseline"]["cost"] / row["candidate"]["cost"])
                row["selected"] = True
                census["strict_lower_selected"] += 1
    rows.sort(
        key=lambda row: (
            not bool(row.get("selected")),
            int(row["task"]),
            int(row.get("component_ordinal", row.get("removal_ordinal", 0))),
        )
    )
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
        "method": {
            "closed_dtype_components": True,
            "exact_integer_interval_proof": True,
            "input_output_dtype_changes": False,
            "shape_cloak": False,
            "approximate_float_narrowing": False,
            "targets": {dtype_name(source): [dtype_name(target) for target in targets] for source, targets in TARGETS.items()},
        },
        "census": dict(census),
        "strict_inference_authority_failures": strict_failures,
        "rows": rows,
        "errors": errors,
        "selected": [row for row in rows if row.get("selected")],
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "census": dict(census),
                "strict_inference_authority_failures": len(strict_failures),
                "tested_rows": len(rows),
                "errors": len(errors),
                "selected": [
                    {key: row.get(key) for key in ("task", "source_dtype", "target_dtype", "baseline", "candidate", "path", "sha256", "projected_gain")}
                    for row in rows if row.get("selected")
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
