#!/usr/bin/env python3
"""All-400 global scalar-carrier elimination scan.

An initializer is removed only when every one of its node-input uses has a
supported proof-preserving rewrite.  The scan emits both per-initializer and
task-wide combined candidates, profiles them with the official cost function,
and retains only strictly lower full/strict models.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


UNSIGNED = {
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
}
NEG_TYPES = {
    TensorProto.FLOAT16,
    TensorProto.FLOAT,
    TensorProto.DOUBLE,
    TensorProto.BFLOAT16,
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
}

# Previously audited sign/domain proofs.  A source name is part of the proof:
# a positive scalar is not moved to Selu merely because it is positive.
PROVEN_SELU: dict[tuple[int, str], dict[str, Any]] = {
    (13, "half_h"): {
        "mode": "mul", "sources": {"p0x2_h"},
        "proof": "p0x2_h=2*min(marker coordinates)>=0 on valid task013 inputs",
    },
    (66, "ln2"): {
        "mode": "div", "sources": {"selLog"},
        "proof": "retained task066 reachable-domain Div/Selu exhaustive proof",
    },
    (90, "ln2"): {
        "mode": "div", "sources": {"ln_lowbit", "ln_sel_run"},
        "proof": "logs of positive powers of two are finite nonnegative",
    },
    (134, "hInv29"): {
        "mode": "mul", "sources": {"rowq", "col", "m2"},
        "proof": "decoded coordinate/count sources are finite nonnegative",
    },
    (158, "coord_pack"): {
        "mode": "mul", "sources": {"p_row", "q_row"},
        "proof": "gathered anchor rows plus boolean phase are finite nonnegative",
    },
    (205, "rowpow_thr"): {
        "mode": "mul", "sources": {"tall_f", "roww_max"},
        "proof": "boolean-mask sum and maximum of nonnegative contractions are >=0",
    },
    (209, "fhalf16"): {
        "mode": "mul", "sources": {"ysr16", "ysc16"},
        "proof": "one-hot coordinate sums are finite nonnegative",
    },
    (209, "ln2"): {
        "mode": "div", "sources": {"pclog"},
        "proof": "log of a positive low bit is finite nonnegative",
    },
    (233, "qseq_256_f16"): {
        "mode": "mul", "sources": {"qseq_high_f16"},
        "proof": "reachable qseq_high_f16 is nonnegative and never negative zero",
    },
    (245, "two_f16"): {
        "mode": "div", "sources": {"rr_log", "rc_log", "gr_log", "gc_log"},
        "proof": "logs of positive power-of-two row/column codes are nonnegative",
    },
    (366, "safe_name_19"): {
        "mode": "mul", "sources": {"safe_name_40"},
        "proof": "nonnegative one-hot coordinate contraction",
    },
    (366, "safe_name_31"): {
        "mode": "mul", "sources": {"safe_name_752", "safe_name_753"},
        "proof": "nonnegative gathered coordinates and bounded offsets",
    },
}


@dataclass(frozen=True)
class Rewrite:
    node_index: int
    input_index: int
    kind: str
    source: str
    output: str
    output_shape: tuple[int, ...]
    source_shape: tuple[int, ...]
    gamma: float | None = None
    proof: str = ""


@dataclass
class Plan:
    task: int
    initializer: str
    dtype: str
    shape: list[int]
    value: str
    uses: int
    rewrites: list[Rewrite]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"scalar262_{task:03d}_{label}_") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def metadata(model: onnx.ModelProto) -> tuple[dict[str, int], dict[str, tuple[int, ...] | None]]:
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=False, data_prop=True)
    types: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...] | None] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        if not value.type.HasField("tensor_type"):
            continue
        tensor = value.type.tensor_type
        types[value.name] = tensor.elem_type
        if all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in tensor.shape.dim):
            shapes[value.name] = tuple(int(dim.dim_value) for dim in tensor.shape.dim)
        elif len(tensor.shape.dim) == 0:
            shapes[value.name] = ()
        else:
            shapes[value.name] = None
    for item in inferred.graph.initializer:
        types[item.name] = item.data_type
        shapes[item.name] = tuple(int(dim) for dim in item.dims)
    return types, shapes


def scalar_text(array: np.ndarray) -> str:
    value = array.reshape(-1)[0]
    if array.dtype.kind == "f":
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "+inf"
        if np.isneginf(value):
            return "-inf"
        if value == 0:
            return "-0.0" if bool(np.signbit(value)) else "+0.0"
        return repr(float(value))
    return str(int(value))


def exact_value(array: np.ndarray, wanted: int) -> bool:
    if array.size != 1 or array.dtype.kind not in "fiu":
        return False
    value = array.reshape(-1)[0]
    return bool(np.isfinite(value) and value == wanted)


def classify(
    task: int,
    initializer: str,
    array: np.ndarray,
    node_index: int,
    input_index: int,
    node: onnx.NodeProto,
    types: dict[str, int],
    shapes: dict[str, tuple[int, ...] | None],
    initializer_names: set[str],
) -> tuple[Rewrite | None, str]:
    occurrences = [index for index, name in enumerate(node.input) if name == initializer]
    if occurrences != [input_index]:
        return None, "initializer_repeated_in_node"
    if len(node.output) != 1 or not node.output[0]:
        return None, "non_single_output"
    output = node.output[0]
    output_shape = shapes.get(output)
    if output_shape is None:
        return None, "unknown_output_shape"

    kind: str | None = None
    source = ""
    gamma: float | None = None
    proof = ""

    if node.op_type in {"Add", "Sub", "Mul", "Div", "Greater", "GreaterOrEqual"} and len(node.input) == 2:
        source = node.input[1 - input_index]
        if not source or source in initializer_names:
            return None, "other_operand_is_initializer"
        source_shape = shapes.get(source)
        if source_shape is None:
            return None, "unknown_source_shape"

        if node.op_type == "Add" and exact_value(array, 0):
            kind = "identity_add_zero"
        elif node.op_type == "Sub" and exact_value(array, 0):
            if input_index == 1:
                kind = "identity_sub_zero"
            elif types.get(source) in NEG_TYPES:
                kind = "neg_sub_zero_left"
            else:
                return None, "neg_unsupported_dtype"
        elif node.op_type == "Mul" and exact_value(array, 1):
            kind = "identity_mul_one"
        elif node.op_type == "Div" and input_index == 1 and exact_value(array, 1):
            kind = "identity_div_one"
        elif node.op_type == "Mul" and exact_value(array, -1):
            if types.get(source) in NEG_TYPES:
                kind = "neg_mul_minus_one"
            else:
                return None, "neg_unsupported_dtype"
        elif (
            node.op_type == "Greater" and input_index == 1 and exact_value(array, 0)
            and types.get(source) in UNSIGNED
        ):
            kind = "cast_unsigned_gt_zero"
        elif (
            node.op_type == "GreaterOrEqual" and input_index == 1 and exact_value(array, 1)
            and types.get(source) in UNSIGNED
        ):
            kind = "cast_unsigned_ge_one"
        else:
            selu = PROVEN_SELU.get((task, initializer))
            if selu and source in selu["sources"] and np.isfinite(array.reshape(-1)[0]) and array.reshape(-1)[0] > 0:
                if node.op_type == "Mul" and selu["mode"] == "mul":
                    kind = "selu_positive_mul"
                    gamma = float(np.float32(float(array.reshape(-1)[0])))
                    proof = str(selu["proof"])
                elif node.op_type == "Div" and input_index == 1 and selu["mode"] == "div":
                    kind = "selu_positive_div"
                    gamma = float(np.float32(1.0 / float(array.reshape(-1)[0])))
                    proof = str(selu["proof"])
        if kind is None:
            return None, f"unsupported_{node.op_type}_input{input_index}"
        return Rewrite(
            node_index, input_index, kind, source, output,
            output_shape, source_shape, gamma, proof,
        ), ""

    if node.op_type == "Clip" and input_index == 1 and exact_value(array, 0):
        if not node.input or types.get(node.input[0]) not in UNSIGNED:
            return None, "clip_data_not_unsigned"
        return Rewrite(
            node_index, input_index, "omit_unsigned_clip_min_zero", node.input[0], output,
            output_shape, shapes.get(node.input[0]) or (), None,
            "unsigned input is already >= 0",
        ), ""

    return None, f"unsupported_{node.op_type}_input{input_index}"


def plans_for_model(task: int, model: onnx.ModelProto) -> tuple[list[Plan], list[dict[str, Any]]]:
    try:
        types, shapes = metadata(model)
    except Exception as exc:  # noqa: BLE001
        return [], [{"task": task, "metadata_error": f"{type(exc).__name__}: {exc}"}]
    arrays = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    initializer_names = set(arrays)
    uses: dict[str, list[tuple[int, int, onnx.NodeProto]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name in arrays:
                uses[name].append((node_index, input_index, node))

    plans: list[Plan] = []
    census: list[dict[str, Any]] = []
    graph_inputs = {value.name for value in model.graph.input}
    for item in model.graph.initializer:
        array = arrays[item.name]
        if array.size != 1:
            continue
        row: dict[str, Any] = {
            "task": task,
            "initializer": item.name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "value": scalar_text(array),
            "uses": len(uses[item.name]),
            "graph_input": item.name in graph_inputs,
            "use_sites": [],
        }
        blockers = []
        rewrites = []
        if not uses[item.name]:
            blockers.append("unused_initializer_out_of_scope")
        if item.name in graph_inputs:
            blockers.append("initializer_is_graph_input")
        for node_index, input_index, node in uses[item.name]:
            rewrite, blocker = classify(
                task, item.name, array, node_index, input_index, node,
                types, shapes, initializer_names,
            )
            site = {
                "node_index": node_index,
                "op": node.op_type,
                "input_index": input_index,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
            if rewrite is None:
                site["blocker"] = blocker
                blockers.append(blocker)
            else:
                site["rewrite"] = rewrite.kind
                site["source_shape"] = list(rewrite.source_shape)
                site["output_shape"] = list(rewrite.output_shape)
                rewrites.append(rewrite)
            row["use_sites"].append(site)
        row["blockers"] = blockers
        row["fully_coverable"] = bool(uses[item.name]) and not blockers and len(rewrites) == len(uses[item.name])
        row["mixed_use"] = len({rewrite.kind for rewrite in rewrites}) > 1 or len({site["op"] for site in row["use_sites"]}) > 1
        census.append(row)
        if row["fully_coverable"]:
            plans.append(Plan(
                task=task,
                initializer=item.name,
                dtype=str(array.dtype),
                shape=list(array.shape),
                value=scalar_text(array),
                uses=len(uses[item.name]),
                rewrites=rewrites,
            ))
    return plans, census


def shape_initializer(
    model: onnx.ModelProto,
    shape: tuple[int, ...],
    cache: dict[tuple[int, ...], str],
) -> str:
    if shape in cache:
        return cache[shape]
    wanted = np.asarray(shape, dtype=np.int64)
    for item in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(item))
        if array.dtype == np.int64 and array.shape == wanted.shape and np.array_equal(array, wanted):
            cache[shape] = item.name
            return item.name
    base = "scalar_elim_shape_" + ("scalar" if not shape else "x".join(map(str, shape)))
    names = {item.name for item in model.graph.initializer}
    name = base
    suffix = 1
    while name in names:
        suffix += 1
        name = f"{base}_{suffix}"
    model.graph.initializer.append(numpy_helper.from_array(wanted, name=name))
    cache[shape] = name
    return name


def rewrite_node(
    model: onnx.ModelProto,
    node: onnx.NodeProto,
    spec: Rewrite,
    shape_cache: dict[tuple[int, ...], str],
) -> list[onnx.NodeProto]:
    if spec.kind == "omit_unsigned_clip_min_zero":
        result = copy.deepcopy(node)
        inputs = list(result.input)
        inputs[1] = ""
        while inputs and not inputs[-1]:
            inputs.pop()
        del result.input[:]
        result.input.extend(inputs)
        return [result]

    if spec.kind.startswith("identity_"):
        unary_op = "Identity"
        attrs: dict[str, Any] = {}
    elif spec.kind.startswith("neg_"):
        unary_op = "Neg"
        attrs = {}
    elif spec.kind.startswith("cast_unsigned_"):
        unary_op = "Cast"
        attrs = {"to": TensorProto.BOOL}
    elif spec.kind.startswith("selu_positive_"):
        unary_op = "Selu"
        attrs = {"alpha": 1.0, "gamma": spec.gamma}
    else:
        raise ValueError(spec.kind)

    if spec.source_shape == spec.output_shape:
        return [helper.make_node(unary_op, [spec.source], [spec.output], name=node.name, **attrs)]

    shape_name = shape_initializer(model, spec.output_shape, shape_cache)
    if unary_op == "Identity":
        return [helper.make_node("Expand", [spec.source, shape_name], [spec.output], name=node.name)]

    temporary = f"{spec.output}__scalar_elim_{unary_op.lower()}"
    unary = helper.make_node(unary_op, [spec.source], [temporary], name=f"{node.name}_scalar_elim", **attrs)
    expand = helper.make_node("Expand", [temporary, shape_name], [spec.output], name=node.name)
    return [unary, expand]


def apply_plans(model: onnx.ModelProto, plans: list[Plan]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    by_node: dict[int, Rewrite] = {}
    for plan in plans:
        for spec in plan.rewrites:
            if spec.node_index in by_node:
                raise ValueError(f"rewrite collision at node {spec.node_index}")
            by_node[spec.node_index] = spec
    shape_cache: dict[tuple[int, ...], str] = {}
    nodes: list[onnx.NodeProto] = []
    for node_index, node in enumerate(result.graph.node):
        spec = by_node.get(node_index)
        nodes.extend(rewrite_node(result, node, spec, shape_cache) if spec else [node])
    del result.graph.node[:]
    result.graph.node.extend(nodes)

    removed = {plan.initializer for plan in plans}
    graph_inputs = {item.name for item in result.graph.input}
    if removed & graph_inputs:
        raise ValueError(f"cannot remove graph inputs: {sorted(removed & graph_inputs)}")
    kept = [item for item in result.graph.initializer if item.name not in removed]
    if len(result.graph.initializer) - len(kept) != len(removed):
        raise ValueError("initializer removal count mismatch")
    del result.graph.initializer[:]
    result.graph.initializer.extend(kept)
    still_used = Counter(name for node in result.graph.node for name in node.input if name)
    leftovers = sorted(name for name in removed if still_used[name])
    if leftovers:
        raise ValueError(f"removed initializers still used: {leftovers}")
    return result


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    row: dict[str, Any] = {}
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, checker_error=f"{type(exc).__name__}: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        nonstatic = []
        for value in [*inferred.graph.value_info, *inferred.graph.output]:
            if value.type.HasField("tensor_type") and any(
                not dim.HasField("dim_value") or dim.dim_value <= 0
                for dim in value.type.tensor_type.shape.dim
            ):
                nonstatic.append(value.name)
        row["strict_data_prop"] = True
        row["nonstatic"] = nonstatic
    except Exception as exc:  # noqa: BLE001
        row.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}", nonstatic=[])
    try:
        findings = check_conv_bias(model)
        row["conv_bias_findings"] = findings
        row["conv_bias_ub0"] = not findings
    except Exception as exc:  # noqa: BLE001
        row.update(conv_bias_ub0=False, conv_bias_error=f"{type(exc).__name__}: {exc}")
    row["pass"] = bool(
        row.get("full_check") and row.get("strict_data_prop")
        and not row.get("nonstatic") and row.get("conv_bias_ub0")
    )
    return row


def candidate_record(
    task: int,
    model: onnx.ModelProto,
    baseline_profile: dict[str, int],
    plans: list[Plan],
    label: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task": task,
        "label": label,
        "initializers": [plan.initializer for plan in plans],
        "plans": [
            {
                "initializer": plan.initializer,
                "dtype": plan.dtype,
                "shape": plan.shape,
                "value": plan.value,
                "uses": plan.uses,
                "rewrites": [asdict(rewrite) for rewrite in plan.rewrites],
            }
            for plan in plans
        ],
        "baseline_profile": baseline_profile,
    }
    try:
        candidate = apply_plans(model, plans)
        record["structure"] = structure(candidate)
        if not record["structure"]["pass"]:
            return record
        current = profile(candidate, task, label)
        record["candidate_profile"] = current
        record["cost_reduction"] = baseline_profile["cost"] - current["cost"]
        record["strict_lower"] = current["cost"] < baseline_profile["cost"]
        if record["strict_lower"]:
            path = CANDIDATES / f"task{task:03d}_{label}.onnx"
            onnx.save(candidate, path)
            record["path"] = str(path.relative_to(ROOT))
            record["sha256"] = digest(path.read_bytes())
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority ZIP hash changed")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    scalar_census: list[dict[str, Any]] = []
    task_summaries = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            plans, census = plans_for_model(task, model)
            scalar_census.extend(census)
            if not plans:
                continue
            baseline = profile(model, task, "authority")
            task_rows = []
            for plan in plans:
                label = "init_" + "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in plan.initializer)
                row = candidate_record(task, model, baseline, [plan], label)
                rows.append(row)
                task_rows.append(row)
            if len(plans) > 1:
                combined = candidate_record(task, model, baseline, plans, "combined")
                rows.append(combined)
                task_rows.append(combined)
            task_summaries.append({
                "task": task,
                "coverable_initializers": [plan.initializer for plan in plans],
                "strict_lower_labels": [row["label"] for row in task_rows if row.get("strict_lower")],
            })
            print(
                json.dumps({
                    "task": task,
                    "coverable": len(plans),
                    "lower": sum(bool(row.get("strict_lower")) for row in task_rows),
                }),
                flush=True,
            )

    blockers = Counter(
        blocker
        for row in scalar_census
        for blocker in row.get("blockers", [])
    )
    lower = [row for row in rows if row.get("strict_lower")]
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks_scanned": len(members),
        "scalar_initializers": len(scalar_census),
        "fully_coverable_initializers": sum(bool(row.get("fully_coverable")) for row in scalar_census),
        "fully_coverable_mixed_use_initializers": sum(
            bool(row.get("fully_coverable") and row.get("mixed_use")) for row in scalar_census
        ),
        "candidate_records": len(rows),
        "strict_lower_records": len(lower),
        "strict_lower_tasks": sorted({row["task"] for row in lower}),
        "blocker_census": dict(sorted(blockers.items())),
        "task_summaries": task_summaries,
        "rows": rows,
        "scalar_census": scalar_census,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: payload[key]
        for key in (
            "tasks_scanned", "scalar_initializers", "fully_coverable_initializers",
            "fully_coverable_mixed_use_initializers", "candidate_records",
            "strict_lower_records", "strict_lower_tasks",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
