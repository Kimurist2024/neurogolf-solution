#!/usr/bin/env python3
"""All-400 ReduceSum/ReduceMean -> scalar arithmetic fusion scan."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


REDUCERS = {"ReduceSum", "ReduceMean"}
ARITHMETIC = {"Mul", "Div", "Add", "Sub"}
FLOAT_DTYPES = {
    onnx.TensorProto.FLOAT16,
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.DOUBLE,
}
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def typed_values(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    return {
        value.name: value
        for value in list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    }


def static_shape(value: onnx.ValueInfoProto | None) -> list[int] | None:
    if value is None:
        return None
    shape = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value < 0:
            return None
        shape.append(int(dim.dim_value))
    return shape


def constant_arrays(model: onnx.ModelProto) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    values = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    kinds = {item.name: "initializer" for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        array = None
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                array = np.asarray(numpy_helper.to_array(attr.t))
            elif attr.name == "value_float":
                array = np.asarray(attr.f, dtype=np.float32)
            elif attr.name == "value_floats":
                array = np.asarray(attr.floats, dtype=np.float32)
            elif attr.name == "value_int":
                array = np.asarray(attr.i, dtype=np.int64)
            elif attr.name == "value_ints":
                array = np.asarray(attr.ints, dtype=np.int64)
        if array is not None:
            values[node.output[0]] = array
            kinds[node.output[0]] = "Constant"
    return values, kinds


def reduction_axes(
    node: onnx.NodeProto,
    constants: dict[str, np.ndarray],
    rank: int,
) -> list[int] | None:
    axes: list[int] | None = None
    if len(node.input) > 1 and node.input[1]:
        array = constants.get(node.input[1])
        if array is None:
            return None
        axes = [int(value) for value in np.asarray(array).reshape(-1)]
    else:
        for attr in node.attribute:
            if attr.name == "axes":
                axes = [int(value) for value in attr.ints]
                break
    noop_empty = next((int(attr.i) for attr in node.attribute if attr.name == "noop_with_empty_axes"), 0)
    if axes is None or (not axes and not noop_empty):
        axes = list(range(rank))
    elif not axes and noop_empty:
        axes = []
    normalized = sorted({axis + rank if axis < 0 else axis for axis in axes})
    if any(axis < 0 or axis >= rank for axis in normalized):
        return None
    return normalized


def scalar_identity(op: str, reduce_slot: int, value: Any) -> bool:
    if op == "Mul":
        return bool(value == 1)
    if op == "Add":
        return bool(value == 0)
    if op == "Sub":
        return reduce_slot == 0 and bool(value == 0)
    if op == "Div":
        return reduce_slot == 0 and bool(value == 1)
    return False


def einsum_equation(input_rank: int, scalar_rank: int, axes: list[int]) -> str:
    if input_rank + scalar_rank > len(LETTERS):
        raise ValueError("rank exceeds Einsum label supply")
    input_labels = LETTERS[:input_rank]
    scalar_labels = LETTERS[input_rank : input_rank + scalar_rank]
    output_labels = "".join(label for index, label in enumerate(input_labels) if index not in axes)
    return f"{input_labels},{scalar_labels}->{output_labels}"


def pattern_rows(model: onnx.ModelProto, task: int, inferred: onnx.ModelProto | None) -> list[dict[str, Any]]:
    constants, kinds = constant_arrays(model)
    consumers: dict[str, list[tuple[int, int, onnx.NodeProto]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            consumers[name].append((node_index, input_index, node))
    typed = typed_values(inferred) if inferred is not None else {}
    graph_outputs = {value.name for value in model.graph.output}
    rows = []
    for reduce_index, reducer in enumerate(model.graph.node):
        if reducer.op_type not in REDUCERS or not reducer.output:
            continue
        uses = consumers.get(reducer.output[0], [])
        if len(uses) != 1:
            continue
        arithmetic_index, reduce_slot, arithmetic = uses[0]
        if arithmetic.op_type not in ARITHMETIC or len(arithmetic.input) != 2:
            continue
        scalar_name = arithmetic.input[1 - reduce_slot]
        scalar = constants.get(scalar_name)
        if scalar is None or scalar.size != 1:
            continue
        value = scalar.reshape(-1)[0].item()
        input_info = typed.get(reducer.input[0])
        reduce_info = typed.get(reducer.output[0])
        input_shape = static_shape(input_info)
        reduce_shape = static_shape(reduce_info)
        dtype = reduce_info.type.tensor_type.elem_type if reduce_info is not None else 0
        axes = reduction_axes(reducer, constants, len(input_shape)) if input_shape is not None else None
        keepdims = next((int(attr.i) for attr in reducer.attribute if attr.name == "keepdims"), 1)
        identity = scalar_identity(arithmetic.op_type, reduce_slot, value)
        eligible_kind = None
        reason = None
        equation = None
        fused_scale = None
        if identity:
            eligible_kind = "identity_elimination"
        elif (
            keepdims == 0
            and input_shape is not None
            and reduce_shape is not None
            and axes is not None
            and dtype in FLOAT_DTYPES
            and arithmetic.op_type in {"Mul", "Div"}
            and (arithmetic.op_type != "Div" or reduce_slot == 0)
        ):
            count = math.prod(input_shape[axis] for axis in axes)
            if reducer.op_type == "ReduceSum" and arithmetic.op_type == "Mul":
                fused_scale = value
            elif reducer.op_type == "ReduceSum" and arithmetic.op_type == "Div":
                fused_scale = 1.0 / value
            elif reducer.op_type == "ReduceMean" and arithmetic.op_type == "Mul":
                fused_scale = value / count
            elif reducer.op_type == "ReduceMean" and arithmetic.op_type == "Div":
                fused_scale = 1.0 / (value * count)
            equation = einsum_equation(len(input_shape), scalar.ndim, axes)
            eligible_kind = "einsum_scalar_fusion"
        else:
            reason = "no exact one-node identity/Einsum construction under static dtype/shape gates"
        rows.append(
            {
                "task": task,
                "reduce_node_index": reduce_index,
                "arithmetic_node_index": arithmetic_index,
                "reduce_op": reducer.op_type,
                "arithmetic_op": arithmetic.op_type,
                "reduce_input_slot": reduce_slot,
                "reduce_output": reducer.output[0],
                "arithmetic_output": arithmetic.output[0],
                "arithmetic_output_is_graph_output": arithmetic.output[0] in graph_outputs,
                "scalar_name": scalar_name,
                "scalar_kind": kinds[scalar_name],
                "scalar_shape": list(scalar.shape),
                "scalar_value": value,
                "dtype": onnx.TensorProto.DataType.Name(dtype) if dtype else "UNKNOWN",
                "input_shape": input_shape,
                "reduce_shape": reduce_shape,
                "axes": axes,
                "keepdims": keepdims,
                "identity": identity,
                "eligible_kind": eligible_kind,
                "equation": equation,
                "fused_scale": fused_scale,
                "ineligible_reason": reason,
            }
        )
    return rows


def build_candidate(model: onnx.ModelProto, row: dict[str, Any]) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    reduce_index = int(row["reduce_node_index"])
    arithmetic_index = int(row["arithmetic_node_index"])
    final_output = str(row["arithmetic_output"])
    old_reduce_output = str(row["reduce_output"])
    nodes: list[onnx.NodeProto] = []
    for node_index, node in enumerate(candidate.graph.node):
        if node_index == arithmetic_index:
            continue
        if node_index == reduce_index:
            if row["eligible_kind"] == "identity_elimination":
                node.output[0] = final_output
                nodes.append(node)
            elif row["eligible_kind"] == "einsum_scalar_fusion":
                nodes.append(
                    helper.make_node(
                        "Einsum",
                        [node.input[0], str(row["scalar_name"])],
                        [final_output],
                        equation=str(row["equation"]),
                        name=f"fused_{node.name or row['reduce_node_index']}_{row['arithmetic_op']}",
                    )
                )
            else:
                raise ValueError("row has no constructible fusion")
        else:
            nodes.append(node)
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    values = [value for value in candidate.graph.value_info if value.name != old_reduce_output]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(values)
    return candidate


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"reduce248_{task:03d}_{label}_") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def session_smoke(model: onnx.ModelProto) -> None:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    census = Counter()
    rows: list[dict[str, Any]] = []
    strict_failures = []
    errors = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        census["authority_models"] = len(members)
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            source_bytes = archive.read(member)
            model = onnx.load_from_string(source_bytes)
            try:
                inferred = onnx.shape_inference.infer_shapes(
                    copy.deepcopy(model), strict_mode=True, data_prop=True
                )
            except Exception as exc:  # authority-side failure, still raw-census the graph
                inferred = None
                strict_failures.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
            found = pattern_rows(model, task, inferred)
            census["detected_patterns"] += len(found)
            if found:
                census["tasks_with_patterns"] += 1
            baseline = profile(model, task, "authority") if any(row["eligible_kind"] for row in found) else None
            best: tuple[int, onnx.ModelProto, dict[str, Any]] | None = None
            for row in found:
                census[f"reduce_{row['reduce_op'].lower()}"] += 1
                census[f"arithmetic_{row['arithmetic_op'].lower()}"] += 1
                if row["eligible_kind"] is None:
                    rows.append(row)
                    continue
                census[f"eligible_{row['eligible_kind']}"] += 1
                try:
                    candidate = build_candidate(model, row)
                    onnx.checker.check_model(copy.deepcopy(candidate), full_check=True)
                    onnx.shape_inference.infer_shapes(
                        copy.deepcopy(candidate), strict_mode=True, data_prop=True
                    )
                    session_smoke(candidate)
                    current = profile(candidate, task, row["eligible_kind"])
                    row.update(
                        {
                            "authority_member_sha256": hashlib.sha256(source_bytes).hexdigest(),
                            "baseline": baseline,
                            "candidate": current,
                            "full_check": True,
                            "strict_shape_inference_data_prop": True,
                            "ort_disable_all_session": True,
                            "memory_nonincrease": current["memory"] <= baseline["memory"],
                            "strict_lower": current["cost"] < baseline["cost"],
                        }
                    )
                    if row["strict_lower"] and row["memory_nonincrease"]:
                        saving = baseline["cost"] - current["cost"]
                        if best is None or saving > best[0]:
                            best = (saving, candidate, row)
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    errors.append(copy.deepcopy(row))
                rows.append(row)
            if best is not None:
                _, candidate, row = best
                path = CANDIDATES / f"task{task:03d}_reduce_scalar_fusion.onnx"
                onnx.save(candidate, path)
                row["preaudit_selected"] = True
                row["path"] = str(path.relative_to(ROOT))
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                row["projected_gain"] = math.log(row["baseline"]["cost"] / row["candidate"]["cost"])
                census["strict_lower_pre_audit"] += 1

    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
        "method": {
            "all_400_raw_graph_census": True,
            "single_consumer_required": True,
            "constant_scalar_required": True,
            "constructors": ["identity_elimination", "Einsum reduction+scalar"],
            "input_output_dtype_changes": False,
            "shape_cloak": False,
        },
        "census": dict(census),
        "strict_inference_authority_failures": strict_failures,
        "rows": rows,
        "errors": errors,
        "preaudit_selected": [row for row in rows if row.get("preaudit_selected")],
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"census": dict(census), "rows": rows, "errors": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
