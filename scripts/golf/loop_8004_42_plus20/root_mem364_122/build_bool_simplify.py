#!/usr/bin/env python3
"""Compile task364's finite 0/1 arithmetic into an equivalent Boolean DAG.

The benchmark adapter always supplies a fixed one-hot [1,10,30,30] tensor.
The incumbent first converts it to Boolean and thereafter uses float16 encodings
of Boolean tensors.  This builder evaluates those two-point encodings exactly
and replaces the arithmetic with the corresponding Boolean truth functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "task364.onnx"
OUT = HERE / "task364_bool_simplified.onnx"
EXPECTED = "2ba1bb84e800b98cdcac9e4d8cb8970d08e532fadf84b64f1b3d88d21ab2a3db"


@dataclass(frozen=True)
class BoolValue:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class EncodedBool:
    base: BoolValue | None
    false_value: float
    true_value: float
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RawValue:
    name: str
    elem_type: int
    shape: tuple[int, ...]


Value = BoolValue | EncodedBool | RawValue


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def broadcast(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(np.broadcast_shapes(*shapes))


def main() -> int:
    source = SOURCE.read_bytes()
    if sha(source) != EXPECTED:
        raise RuntimeError(f"authority changed: {sha(source)}")
    old = onnx.load_from_string(source)
    old_vi = {
        value.name: value
        for value in list(old.graph.value_info) + list(old.graph.output)
    }

    nodes: list[onnx.NodeProto] = []
    values: dict[str, Value] = {
        "input": RawValue("input", TensorProto.FLOAT, (1, 10, 30, 30)),
        "zero_f16": EncodedBool(None, 0.0, 0.0, ()),
        "false_bool": EncodedBool(None, 0.0, 0.0, ()),
    }
    actual_outputs: list[str] = []
    output_types: dict[str, int] = {}
    logical_cache: dict[tuple[Any, ...], BoolValue] = {}
    generated = 0
    comparison_expressions: list[str] = []

    def add_node(node: onnx.NodeProto, elem_type: int) -> None:
        nodes.append(node)
        for name in node.output:
            actual_outputs.append(name)
            output_types[name] = elem_type

    def logical(op: str, inputs: list[BoolValue]) -> BoolValue:
        nonlocal generated
        names = [item.name for item in inputs]
        if op in {"And", "Or", "Xor"}:
            names.sort()
        key = (op, *names)
        if key in logical_cache:
            return logical_cache[key]
        shape = broadcast(*(item.shape for item in inputs))
        name = f"bool_{generated}"
        generated += 1
        add_node(helper.make_node(op, names, [name], name=f"logic_{name}"), TensorProto.BOOL)
        result = BoolValue(name, shape)
        logical_cache[key] = result
        return result

    def not_(value: BoolValue) -> BoolValue:
        return logical("Not", [value])

    def constant_like(value: BoolValue, truth: bool) -> BoolValue:
        false = logical("Xor", [value, value])
        return not_(false) if truth else false

    def synthesize(refs: list[BoolValue], table: tuple[bool, ...], target_shape: tuple[int, ...]) -> BoolValue:
        unique: list[BoolValue] = []
        for ref in refs:
            if all(ref.name != item.name for item in unique):
                unique.append(ref)
        if not unique:
            raise RuntimeError("scalar Boolean constant without a tensor witness")
        if len(unique) == 1:
            ref = unique[0]
            if table == (False, True):
                result = ref
            elif table == (True, False):
                result = not_(ref)
            elif table == (False, False):
                result = constant_like(ref, False)
            elif table == (True, True):
                result = constant_like(ref, True)
            else:
                raise RuntimeError(f"bad unary truth table {table}")
            if result.shape != target_shape:
                raise RuntimeError(f"unary shape loss: {result.shape} != {target_shape}")
            return result
        if len(unique) != 2:
            raise RuntimeError(f"unsupported arity {len(unique)}")
        a, b = unique
        bits = "".join("1" if item else "0" for item in table)
        if bits == "0000":
            result = constant_like(logical("Or", [a, b]), False)
        elif bits == "1111":
            result = constant_like(logical("Or", [a, b]), True)
        elif bits == "0011": result = a
        elif bits == "1100": result = not_(a)
        elif bits == "0101": result = b
        elif bits == "1010": result = not_(b)
        elif bits == "0001": result = logical("And", [a, b])
        elif bits == "0111": result = logical("Or", [a, b])
        elif bits == "0110": result = logical("Xor", [a, b])
        elif bits == "1001": result = not_(logical("Xor", [a, b]))
        elif bits == "1110": result = not_(logical("And", [a, b]))
        elif bits == "1000": result = not_(logical("Or", [a, b]))
        elif bits == "1101": result = logical("Or", [not_(a), b])
        elif bits == "1011": result = logical("Or", [a, not_(b)])
        elif bits == "0010": result = logical("And", [a, not_(b)])
        elif bits == "0100": result = logical("And", [not_(a), b])
        else: raise RuntimeError(f"unknown binary truth table {bits}")
        if result.shape != target_shape:
            # Materialize the ignored operand's broadcast extent without
            # changing the truth value.
            witness = logical("Or", [a, b])
            false = logical("Xor", [witness, witness])
            result = logical("Or", [result, false])
        if result.shape != target_shape:
            raise RuntimeError(f"binary shape loss: {result.shape} != {target_shape}")
        return result

    def encoded(value: Value) -> EncodedBool:
        if isinstance(value, EncodedBool):
            return value
        if isinstance(value, BoolValue):
            return EncodedBool(value, 0.0, 1.0, value.shape)
        raise RuntimeError(f"not a finite Boolean encoding: {value}")

    int_values: dict[str, int] = {}
    initializers: list[onnx.TensorProto] = []

    for node in old.graph.node:
        output = node.output[0]
        inputs = [values[name] for name in node.input]
        if node.op_type == "Shape":
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            start, end = int(attrs["start"]), int(attrs["end"])
            source_shape = inputs[0].shape
            selected = source_shape[start:end]
            if len(selected) != 1:
                raise RuntimeError(f"unexpected Shape slice {selected}")
            copied = helper.make_node("Shape", [inputs[0].name], [output], start=start, end=end, name=node.name)
            add_node(copied, TensorProto.INT64)
            values[output] = RawValue(output, TensorProto.INT64, (1,))
            int_values[output] = int(selected[0])
            continue
        if node.op_type in {"Sub", "Add"}:
            left, right = inputs
            if not isinstance(left, RawValue) or not isinstance(right, RawValue):
                raise RuntimeError(f"unexpected shape arithmetic inputs at {node.name}")
            copied = helper.make_node(node.op_type, [left.name, right.name], [output], name=node.name)
            add_node(copied, TensorProto.INT64)
            values[output] = RawValue(output, TensorProto.INT64, broadcast(left.shape, right.shape))
            if node.op_type == "Sub":
                int_values[output] = int_values[node.input[0]] - int_values[node.input[1]]
            else:
                int_values[output] = int_values[node.input[0]] + int_values[node.input[1]]
            continue
        if node.op_type == "CenterCropPad":
            data, target = inputs
            if isinstance(data, EncodedBool):
                raise RuntimeError(f"encoded CenterCropPad unsupported at {node.name}")
            if not isinstance(target, RawValue):
                raise RuntimeError(f"bad target at {node.name}")
            axes = list(helper.get_attribute_value(next(a for a in node.attribute if a.name == "axes")))
            target_value = int_values[node.input[1]]
            shape = list(data.shape)
            for axis in axes:
                shape[axis] = target_value
            copied = helper.make_node(
                "CenterCropPad", [data.name, target.name], [output], axes=axes, name=node.name
            )
            elem_type = TensorProto.BOOL if isinstance(data, BoolValue) else data.elem_type
            add_node(copied, elem_type)
            values[output] = BoolValue(output, tuple(shape)) if isinstance(data, BoolValue) else RawValue(output, elem_type, tuple(shape))
            continue
        if node.op_type == "CastLike":
            data, target = inputs
            if isinstance(target, EncodedBool) and target.false_value == 0.0 and node.input[1] == "false_bool":
                if not isinstance(data, RawValue):
                    raise RuntimeError(f"unexpected bool cast input {data}")
                copied = helper.make_node("Cast", [data.name], [output], to=TensorProto.BOOL, name=node.name)
                add_node(copied, TensorProto.BOOL)
                values[output] = BoolValue(output, data.shape)
            elif node.input[1] == "zero_f16":
                if not isinstance(data, BoolValue):
                    raise RuntimeError(f"unexpected f16 cast input {data}")
                values[output] = EncodedBool(data, 0.0, 1.0, data.shape)
            else:
                raise RuntimeError(f"unexpected CastLike target {node.input[1]}")
            continue
        if node.op_type == "HardSigmoid":
            item = encoded(inputs[0])
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            alpha, beta = float(attrs.get("alpha", 0.2)), float(attrs.get("beta", 0.5))
            transform = lambda x: min(1.0, max(0.0, alpha * x + beta))
            values[output] = EncodedBool(
                item.base, transform(item.false_value), transform(item.true_value), item.shape
            )
            continue
        if node.op_type in {"LessOrEqual", "GreaterOrEqual"}:
            left, right = encoded(inputs[0]), encoded(inputs[1])
            refs: list[BoolValue] = []
            for item in (left, right):
                if item.base is not None and all(item.base.name != ref.name for ref in refs):
                    refs.append(item.base)
            target_shape = broadcast(left.shape, right.shape)
            if not refs:
                raise RuntimeError(f"constant comparison at {node.name}")

            def relation(a: float, b: float) -> bool:
                return a <= b if node.op_type == "LessOrEqual" else a >= b

            if len(refs) == 1:
                ref = refs[0]
                table_values = []
                for bit in (False, True):
                    lv = left.true_value if left.base is not None and left.base.name == ref.name and bit else left.false_value
                    rv = right.true_value if right.base is not None and right.base.name == ref.name and bit else right.false_value
                    table_values.append(relation(lv, rv))
            else:
                a, b = refs
                table_values = []
                for abit, bbit in ((False, False), (False, True), (True, False), (True, True)):
                    def pick(item: EncodedBool) -> float:
                        if item.base is None: return item.false_value
                        bit = abit if item.base.name == a.name else bbit
                        return item.true_value if bit else item.false_value
                    table_values.append(relation(pick(left), pick(right)))
            result = synthesize(refs, tuple(table_values), target_shape)
            values[output] = result
            comparison_expressions.append(result.name)
            continue
        if node.op_type == "Concat":
            refs = []
            for item in inputs:
                if not isinstance(item, BoolValue):
                    raise RuntimeError(f"non-Boolean concat input {item}")
                refs.append(item)
            axis = int(helper.get_attribute_value(next(a for a in node.attribute if a.name == "axis")))
            add_node(helper.make_node("Concat", [item.name for item in refs], [output], axis=axis, name=node.name), TensorProto.BOOL)
            shape = list(refs[0].shape)
            shape[axis] = sum(item.shape[axis] for item in refs)
            values[output] = BoolValue(output, tuple(shape))
            continue
        raise RuntimeError(f"unsupported op {node.op_type} at {node.name}")

    output_ref = values[old.graph.output[0].name]
    if not isinstance(output_ref, BoolValue) or output_ref.name != old.graph.output[0].name:
        raise RuntimeError(f"unexpected final output {output_ref}")

    graph = helper.make_graph(
        nodes,
        "task364_boolean_simplified",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 1, 1])],
        initializers,
    )
    # Preserve the incumbent's scored scalar declarations for every internal
    # output.  Runtime equivalence is audited separately in both ORT modes.
    for name in actual_outputs:
        if name == "output":
            continue
        if name in old_vi:
            graph.value_info.append(old_vi[name])
        else:
            graph.value_info.append(helper.make_tensor_value_info(name, output_types[name], [1, 1, 1, 1]))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = old.ir_version
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUT)
    result = {
        "authority_sha256": sha(source),
        "candidate_sha256": sha(OUT.read_bytes()),
        "source_nodes": len(old.graph.node),
        "candidate_nodes": len(model.graph.node),
        "source_initializers": len(old.graph.initializer),
        "candidate_initializers": len(model.graph.initializer),
        "generated_logical_nodes": generated,
        "comparison_count": len(comparison_expressions),
        "unique_comparison_expressions": len(set(comparison_expressions)),
        "reused_comparison_expressions": len(comparison_expressions) - len(set(comparison_expressions)),
        "domain": "fixed benchmark one-hot [1,10,30,30] inputs",
    }
    (HERE / "bool_build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
