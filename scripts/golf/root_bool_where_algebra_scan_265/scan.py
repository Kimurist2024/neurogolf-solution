#!/usr/bin/env python3
"""Compile Boolean Where branches into exact And/Or/Not/Expand forms."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"boolwhere265_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def metadata(model: onnx.ModelProto) -> tuple[dict[str, int], dict[str, tuple[int, ...] | None]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    types: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...] | None] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor = value.type.tensor_type
        types[value.name] = tensor.elem_type
        dims = tensor.shape.dim
        shapes[value.name] = (
            tuple(int(dim.dim_value) for dim in dims)
            if all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims)
            else None
        )
    for init in inferred.graph.initializer:
        types[init.name] = init.data_type
        shapes[init.name] = tuple(int(dim) for dim in init.dims)
    return types, shapes


def uniform_bool(array: np.ndarray | None) -> bool | None:
    if array is None or array.dtype != np.bool_ or array.size == 0:
        return None
    value = bool(array.reshape(-1)[0])
    return value if bool(np.all(array == value)) else None


def prune(model: onnx.ModelProto) -> tuple[list[str], list[str]]:
    graph_outputs = {value.name for value in model.graph.output}
    removed_nodes: list[str] = []
    while True:
        uses = Counter(name for node in model.graph.node for name in node.input if name)
        keep = []
        current = []
        for node in model.graph.node:
            outputs = [name for name in node.output if name]
            if outputs and all(uses[name] == 0 and name not in graph_outputs for name in outputs):
                current.append(node.name or node.op_type)
            else:
                keep.append(node)
        if not current:
            break
        removed_nodes.extend(current)
        del model.graph.node[:]
        model.graph.node.extend(keep)
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    graph_inputs = {value.name for value in model.graph.input}
    removed_inits = [
        item.name for item in model.graph.initializer
        if uses[item.name] == 0 and item.name not in graph_inputs
    ]
    keep_inits = [item for item in model.graph.initializer if item.name not in removed_inits]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_inits)
    return removed_nodes, removed_inits


def direct_or_expand(
    source: str,
    output: str,
    node_name: str,
    shapes: dict[str, tuple[int, ...] | None],
) -> tuple[list[onnx.NodeProto], onnx.TensorProto | None, str] | None:
    if shapes.get(source) == shapes.get(output):
        return [helper.make_node("Identity", [source], [output], name=node_name)], None, "Identity"
    output_shape = shapes.get(output)
    if output_shape is None:
        return None
    shape_name = f"{output}__shape"
    init = numpy_helper.from_array(np.asarray(output_shape, dtype=np.int64), name=shape_name)
    return [helper.make_node("Expand", [source, shape_name], [output], name=node_name)], init, "Expand"


def compile_where(
    node: onnx.NodeProto,
    arrays: dict[str, np.ndarray],
    types: dict[str, int],
    shapes: dict[str, tuple[int, ...] | None],
) -> tuple[list[onnx.NodeProto], onnx.TensorProto | None, str] | None:
    if node.op_type != "Where" or len(node.input) != 3:
        return None
    cond, yes, no = node.input
    if any(types.get(name) != TensorProto.BOOL for name in (cond, yes, no)):
        return None
    ybit, nbit = uniform_bool(arrays.get(yes)), uniform_bool(arrays.get(no))
    output = node.output[0]
    if yes == no:
        return direct_or_expand(yes, output, node.name, shapes)
    if ybit is True and nbit is False:
        return direct_or_expand(cond, output, node.name, shapes)
    if ybit is False and nbit is True:
        if shapes.get(cond) != shapes.get(output):
            expanded = f"{output}__condition"
            output_shape = shapes.get(output)
            if output_shape is None:
                return None
            shape_name = f"{output}__shape"
            init = numpy_helper.from_array(np.asarray(output_shape, dtype=np.int64), name=shape_name)
            nodes = [
                helper.make_node("Expand", [cond, shape_name], [expanded]),
                helper.make_node("Not", [expanded], [output], name=node.name),
            ]
            return nodes, init, "Not(Expand(cond))"
        return [helper.make_node("Not", [cond], [output], name=node.name)], None, "Not(cond)"
    if nbit is False:
        return [helper.make_node("And", [cond, yes], [output], name=node.name)], None, "And(cond,yes)"
    if ybit is True:
        return [helper.make_node("Or", [cond, no], [output], name=node.name)], None, "Or(cond,no)"
    if nbit is True:
        inv = f"{output}__not_condition"
        return [
            helper.make_node("Not", [cond], [inv]),
            helper.make_node("Or", [inv, yes], [output], name=node.name),
        ], None, "Or(Not(cond),yes)"
    if ybit is False:
        inv = f"{output}__not_condition"
        return [
            helper.make_node("Not", [cond], [inv]),
            helper.make_node("And", [inv, no], [output], name=node.name),
        ], None, "And(Not(cond),no)"
    return None


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    census = {"Where": 0, "boolean_where": 0, "patterns": 0}
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            try:
                types, shapes = metadata(model)
            except Exception:
                types, shapes = {}, {}
            baseline = None
            for index, node in enumerate(model.graph.node):
                if node.op_type != "Where":
                    continue
                census["Where"] += 1
                if len(node.input) == 3 and all(types.get(name) == TensorProto.BOOL for name in node.input):
                    census["boolean_where"] += 1
                found = compile_where(node, arrays, types, shapes)
                if found is None:
                    continue
                census["patterns"] += 1
                replacements, shape_init, description = found
                candidate = copy.deepcopy(model)
                nodes = list(candidate.graph.node)
                nodes[index:index + 1] = replacements
                del candidate.graph.node[:]
                candidate.graph.node.extend(nodes)
                if shape_init is not None:
                    candidate.graph.initializer.append(shape_init)
                removed_nodes, removed_inits = prune(candidate)
                record: dict = {
                    "task": task,
                    "node_index": index,
                    "rewrite": description,
                    "removed_nodes": removed_nodes,
                    "removed_initializers": removed_inits,
                }
                try:
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    if baseline is None:
                        baseline = profile(model, task)
                    current = profile(candidate, task)
                    record["baseline"] = baseline
                    record["candidate"] = current
                    record["strict_lower"] = current["cost"] < baseline["cost"]
                    if record["strict_lower"]:
                        path = CANDIDATES / f"task{task:03d}_{index:04d}.onnx"
                        onnx.save(candidate, path)
                        record["path"] = str(path.relative_to(REPO))
                        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(record)
    result = {
        "authority": str(AUTHORITY),
        "tasks": len(members),
        "census": census,
        "strict_lower": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"census": census, "strict_lower": result["strict_lower"]}, indent=2))


if __name__ == "__main__":
    main()
