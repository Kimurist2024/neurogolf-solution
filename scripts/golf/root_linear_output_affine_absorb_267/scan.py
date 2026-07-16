#!/usr/bin/env python3
"""All-400 single-use linear-output scalar/affine absorption scan."""

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
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CANDIDATES = HERE / "candidates"
LINEAR_OPS = {
    "Einsum", "MatMul", "Gemm", "Conv", "ConvInteger",
    "MatMulInteger", "QLinearConv", "QLinearMatMul",
}
ARITHMETIC = {"Mul", "Div", "Add", "Sub"}
KNOWN_SHAPE_CLOAK = {54, 367}

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"linear267_{task:03d}_{label}_") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def metadata(model: onnx.ModelProto) -> tuple[dict[str, tuple[int, ...] | None], dict[str, int]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=False, data_prop=True)
    except Exception:  # noqa: BLE001
        inferred = model
    shapes: dict[str, tuple[int, ...] | None] = {}
    types: dict[str, int] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        if not value.type.HasField("tensor_type"):
            continue
        tensor = value.type.tensor_type
        types[value.name] = tensor.elem_type
        if len(tensor.shape.dim) == 0:
            shapes[value.name] = ()
        elif all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in tensor.shape.dim):
            shapes[value.name] = tuple(int(dim.dim_value) for dim in tensor.shape.dim)
        else:
            shapes[value.name] = None
    for item in inferred.graph.initializer:
        shapes[item.name] = tuple(int(dim) for dim in item.dims)
        types[item.name] = item.data_type
    return shapes, types


def equation(node: onnx.NodeProto) -> str | None:
    for attribute in node.attribute:
        if attribute.name == "equation":
            value = helper.get_attribute_value(attribute)
            return value.decode() if isinstance(value, bytes) else str(value)
    return None


def constant_value(
    name: str,
    initializers: dict[str, np.ndarray],
    constant_nodes: dict[str, np.ndarray],
) -> np.ndarray | None:
    return initializers.get(name, constant_nodes.get(name))


def constant_nodes(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    result = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        if "value" in attributes:
            result[node.output[0]] = np.asarray(numpy_helper.to_array(attributes["value"]))
        elif "value_float" in attributes:
            result[node.output[0]] = np.asarray(attributes["value_float"], dtype=np.float32)
        elif "value_int" in attributes:
            result[node.output[0]] = np.asarray(attributes["value_int"], dtype=np.int64)
    return result


def is_scalar_shape(shape: tuple[int, ...] | None) -> bool:
    return shape is not None and math.prod(shape) == 1


def infer_scalar_term(
    output_labels: str,
    output_shape: tuple[int, ...] | None,
    scalar_shape: tuple[int, ...] | None,
) -> str | None:
    if output_shape is None or scalar_shape is None or len(output_labels) != len(output_shape):
        return None
    if scalar_shape == ():
        return ""
    if math.prod(scalar_shape) != 1:
        return None
    # A rank-r all-one tensor may bind to the rightmost r output labels.  Every
    # current eligible site is rank one with an output dimension of one.
    if len(scalar_shape) > len(output_shape):
        return None
    labels = output_labels[-len(scalar_shape):]
    dims = output_shape[-len(scalar_shape):]
    if any(source != 1 or target != 1 for source, target in zip(scalar_shape, dims)):
        return None
    return labels


def site_reason(
    task: int,
    producer: onnx.NodeProto,
    consumer: onnx.NodeProto,
    linear_side: int,
    other: str,
    other_shape: tuple[int, ...] | None,
    other_constant: np.ndarray | None,
) -> tuple[str, str]:
    if not is_scalar_shape(other_shape):
        return "ineligible", "other_operand_not_scalar"
    if producer.op_type == "Einsum" and consumer.op_type == "Mul":
        return "einsum_operand_fuse", "multiplicative scalar can be appended as an Einsum operand"
    if consumer.op_type == "Div" and linear_side == 1:
        return "ineligible", "scalar_divided_by_linear_output_is_nonlinear_reciprocal"
    if other_constant is None:
        return "ineligible", f"dynamic_{consumer.op_type.lower()}_has_no_supported_linear_bias_or_scale_slot"
    if producer.op_type in {"QLinearConv", "QLinearMatMul"}:
        return "ineligible", "post_quantization_arithmetic_cannot_be_moved across_round_clip_without_proof"
    if producer.op_type in {"Einsum", "MatMul", "Conv", "ConvInteger", "MatMulInteger", "Gemm"}:
        return "offline_constant_probe", "constant scalar may be absorbed if a constant coefficient round-trips"
    return "ineligible", "unsupported_linear_family"


def enumerate_sites(task: int, model: onnx.ModelProto) -> list[dict[str, Any]]:
    shapes, types = metadata(model)
    initializers = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    constants = constant_nodes(model)
    producer_of = {output: (index, node) for index, node in enumerate(model.graph.node) for output in node.output}
    uses: defaultdict[str, list[tuple[int, int, onnx.NodeProto]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                uses[name].append((node_index, input_index, node))
    rows = []
    for producer_index, producer in enumerate(model.graph.node):
        if producer.op_type not in LINEAR_OPS:
            continue
        for output in producer.output:
            consumers = uses[output]
            if len(consumers) != 1:
                continue
            consumer_index, linear_side, consumer = consumers[0]
            if consumer.op_type not in ARITHMETIC or len(consumer.input) != 2 or len(consumer.output) != 1:
                continue
            other = consumer.input[1 - linear_side]
            array = constant_value(other, initializers, constants)
            eligibility, reason = site_reason(
                task, producer, consumer, linear_side, other,
                shapes.get(other), array,
            )
            rows.append({
                "task": task,
                "producer_index": producer_index,
                "producer_op": producer.op_type,
                "producer_inputs": list(producer.input),
                "producer_output": output,
                "producer_output_shape": list(shapes[output]) if shapes.get(output) is not None else None,
                "producer_output_type": types.get(output),
                "consumer_index": consumer_index,
                "consumer_op": consumer.op_type,
                "linear_side": linear_side,
                "other_operand": other,
                "other_shape": list(shapes[other]) if shapes.get(other) is not None else None,
                "other_source": (
                    "initializer" if other in initializers else
                    "Constant" if other in constants else
                    producer_of[other][1].op_type if other in producer_of else "graph_input_or_unknown"
                ),
                "other_constant_dtype": str(array.dtype) if array is not None else None,
                "other_constant_shape": list(array.shape) if array is not None else None,
                "other_constant_uniform": bool(array is not None and array.size and np.all(array == array.reshape(-1)[0])),
                "other_constant_value": (
                    repr(array.reshape(-1)[0].item())
                    if array is not None and array.size and np.all(array == array.reshape(-1)[0]) else None
                ),
                "consumer_output": consumer.output[0],
                "consumer_output_shape": (
                    list(shapes[consumer.output[0]])
                    if shapes.get(consumer.output[0]) is not None else None
                ),
                "eligibility": eligibility,
                "reason": reason,
                "known_shape_cloak": task in KNOWN_SHAPE_CLOAK,
            })
    return rows


def fuse_einsum_mul(model: onnx.ModelProto, site: dict[str, Any]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    producer = result.graph.node[site["producer_index"]]
    consumer = result.graph.node[site["consumer_index"]]
    text = equation(producer)
    if not text or "->" not in text:
        raise ValueError("Einsum equation is not explicit")
    left, output_labels = text.split("->", 1)
    output_shape = tuple(site["producer_output_shape"] or ())
    other_shape = tuple(site["other_shape"] or ())
    term = infer_scalar_term(output_labels, output_shape, other_shape)
    if term is None:
        raise ValueError("cannot map scalar shape to Einsum output labels")
    new_equation = f"{left},{term}->{output_labels}"
    for attribute in producer.attribute:
        if attribute.name == "equation":
            attribute.s = new_equation.encode()
            break
    else:
        raise ValueError("missing equation attribute")
    producer.input.append(site["other_operand"])
    producer.output[0] = site["consumer_output"]
    nodes = [node for index, node in enumerate(result.graph.node) if index != site["consumer_index"]]
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    # Remove stale metadata for the eliminated producer output.  The consumer
    # output metadata remains authoritative for the fused node.
    values = [item for item in result.graph.value_info if item.name != site["producer_output"]]
    del result.graph.value_info[:]
    result.graph.value_info.extend(values)
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
        row["conv_bias_findings"] = check_conv_bias(model)
        row["conv_bias_ub0"] = not row["conv_bias_findings"]
    except Exception as exc:  # noqa: BLE001
        row.update(conv_bias_ub0=False, conv_bias_error=f"{type(exc).__name__}: {exc}")
    row["pass"] = bool(
        row.get("full_check") and row.get("strict_data_prop")
        and not row.get("nonstatic") and row.get("conv_bias_ub0")
    )
    return row


def build_candidate(task: int, model: onnx.ModelProto, site: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"task": task, "site": site}
    try:
        candidate = fuse_einsum_mul(model, site)
        record["structure"] = structure(candidate)
        baseline = profile(model, task, "authority")
        record["baseline_profile"] = baseline
        if record["structure"]["pass"]:
            current = profile(candidate, task, "candidate")
            record["candidate_profile"] = current
            record["activation_bytes_removed"] = baseline["memory"] - current["memory"]
            record["strict_lower"] = current["cost"] < baseline["cost"]
            if record["strict_lower"]:
                path = CANDIDATES / f"task{task:03d}_einsum_scalar_fuse.onnx"
                onnx.save(candidate, path)
                record["path"] = str(path.relative_to(ROOT))
                record["sha256"] = digest(path.read_bytes())
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority ZIP changed")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    sites = []
    candidates = []
    op_census = Counter()
    excluded_fanout = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            current = enumerate_sites(task, model)
            sites.extend(current)
            op_census.update((row["producer_op"], row["consumer_op"]) for row in current)
            for site in current:
                if site["eligibility"] == "einsum_operand_fuse":
                    candidates.append(build_candidate(task, model, site))
            if task == 367:
                uses = Counter(name for node in model.graph.node for name in node.input if name)
                for index, node in enumerate(model.graph.node):
                    if node.op_type == "Einsum" and list(node.output) == ["G"]:
                        excluded_fanout.append({
                            "task": 367,
                            "producer_index": index,
                            "output": "G",
                            "uses": uses["G"],
                            "reason": (
                                "fanout is seven (including Mul and Div), outside the required "
                                "single-use-output gate; task is also known shape-cloaked"
                            ),
                        })
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks_scanned": len(members),
        "linear_to_arithmetic_single_use_sites": len(sites),
        "operator_pair_census": {
            f"{left}->{right}": count
            for (left, right), count in sorted(op_census.items())
        },
        "scalar_operand_sites": sum(is_scalar_shape(tuple(row["other_shape"]) if row["other_shape"] is not None else None) for row in sites),
        "einsum_operand_fuse_sites": sum(row["eligibility"] == "einsum_operand_fuse" for row in sites),
        "offline_constant_probe_sites": sum(row["eligibility"] == "offline_constant_probe" for row in sites),
        "candidate_records": len(candidates),
        "strict_lower_candidates": sum(bool(row.get("strict_lower")) for row in candidates),
        "sites": sites,
        "candidates": candidates,
        "task367_single_use_exclusion": excluded_fanout,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: payload[key]
        for key in (
            "tasks_scanned", "linear_to_arithmetic_single_use_sites",
            "scalar_operand_sites", "einsum_operand_fuse_sites",
            "offline_constant_probe_sites", "candidate_records",
            "strict_lower_candidates",
        )
    }, indent=2))
    for row in candidates:
        print(json.dumps({
            "task": row["task"],
            "structure": row.get("structure"),
            "baseline": row.get("baseline_profile"),
            "candidate": row.get("candidate_profile"),
            "strict_lower": row.get("strict_lower"),
            "error": row.get("error"),
        }), flush=True)


if __name__ == "__main__":
    main()
