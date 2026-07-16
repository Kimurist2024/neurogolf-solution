#!/usr/bin/env python3
"""Fail-closed domain-aware arithmetic identity scan over LB8009.46."""

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
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
OUTPUT = HERE / "scan.json"
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

TARGET_OPS = {
    "Pow", "Mul", "Div", "Add", "Sub", "Abs", "Relu", "LeakyRelu", "Selu",
    "Clip", "Min", "Max", "Less", "LessOrEqual", "Greater", "GreaterOrEqual",
    "Equal", "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
}
UNCONDITIONAL_SINGLETON_REDUCES = {
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
}
UNSIGNED_TYPES = {
    onnx.TensorProto.UINT8,
    onnx.TensorProto.UINT16,
    onnx.TensorProto.UINT32,
    onnx.TensorProto.UINT64,
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def attrs(node: onnx.NodeProto) -> dict[str, Any]:
    result = {}
    for item in node.attribute:
        value = helper.get_attribute_value(item)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        result[item.name] = value
    return result


def value_metadata(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    result = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        if not value.type.HasField("tensor_type"):
            continue
        tensor = value.type.tensor_type
        shape = []
        truthful_static = True
        for dim in tensor.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                truthful_static = False
                shape.append(None)
            else:
                shape.append(int(dim.dim_value))
        result[value.name] = {
            "dtype": int(tensor.elem_type),
            "shape": shape,
            "positive_static_shape": truthful_static,
        }
    return result


def reduction_axes(node: onnx.NodeProto, arrays: dict[str, np.ndarray]) -> list[int] | None:
    if len(node.input) > 1 and node.input[1] in arrays:
        return [int(value) for value in arrays[node.input[1]].reshape(-1)]
    value = attrs(node).get("axes")
    return None if value is None else [int(item) for item in value]


def direct_unsigned_cast_identity(
    node: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if node.op_type not in {"LeakyRelu", "Relu", "Abs"} or len(node.input) != 1:
        return None
    producer = producers.get(node.input[0])
    if producer is None or producer.op_type not in {"Cast", "CastLike"} or not producer.input:
        return None
    source = metadata.get(producer.input[0])
    if source is None or source["dtype"] not in UNSIGNED_TYPES:
        return None
    return {
        "kind": f"{node.op_type.lower()}_unsigned_cast_nonnegative",
        "proof": (
            f"{producer.op_type} source {producer.input[0]} has unsigned ONNX dtype "
            f"{source['dtype']}; its floating result is nonnegative, so {node.op_type}'s "
            "positive branch is the identity"
        ),
    }


def scan_model(task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output if name}
    consumers: dict[str, list[int]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for name in node.input:
            if name:
                consumers[name].append(node_index)

    strict_error = None
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
    except Exception as exc:  # noqa: BLE001
        inferred = model
        strict_error = f"{type(exc).__name__}: {exc}"
    metadata = value_metadata(inferred)
    op_counts: Counter[str] = Counter()
    rows = []
    bypasses = []
    graph_outputs = {value.name for value in model.graph.output}
    duplicate_groups: dict[tuple[str, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for name, array in arrays.items():
        if array.size == 1:
            duplicate_groups[(array.dtype.str, tuple(array.shape), array.tobytes())].append(name)
    deduplications = []
    for names in duplicate_groups.values():
        for removed in names[1:]:
            deduplications.append({
                "kind": "bit_identical_scalar_initializer_alias",
                "kept": names[0],
                "removed": removed,
                "proof": "same dtype, shape, and tensor bytes",
            })

    for node_index, node in enumerate(model.graph.node):
        if node.op_type not in TARGET_OPS:
            continue
        op_counts[node.op_type] += 1
        constant_inputs = []
        for input_index, name in enumerate(node.input):
            if name not in arrays:
                continue
            array = arrays[name]
            constant_inputs.append({
                "input_index": input_index,
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "values": array.reshape(-1).tolist()[:32],
                "elements": int(array.size),
            })
        row: dict[str, Any] = {
            "task": task,
            "node_index": node_index,
            "op": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": attrs(node),
            "constant_inputs": constant_inputs,
        }
        if node.input:
            row["input0_metadata"] = metadata.get(node.input[0])
        if node.output:
            row["output0_metadata"] = metadata.get(node.output[0])

        lead = None
        if (
            strict_error is None
            and len(node.input) >= 1
            and len(node.output) == 1
            and node.output[0] not in graph_outputs
        ):
            source_meta = metadata.get(node.input[0])
            output_meta = metadata.get(node.output[0])
            if node.op_type in UNCONDITIONAL_SINGLETON_REDUCES and source_meta and output_meta:
                axes = reduction_axes(node, arrays)
                keepdims = int(attrs(node).get("keepdims", 1))
                shape = source_meta["shape"]
                if axes is not None and all(dim is not None for dim in shape):
                    normalized = [axis if axis >= 0 else axis + len(shape) for axis in axes]
                    valid = all(0 <= axis < len(shape) for axis in normalized)
                    count = math.prod(shape[axis] for axis in normalized) if valid else 0
                    if count == 1 and keepdims == 1 and shape == output_meta["shape"]:
                        lead = {
                            "kind": "singleton_reduce_identity",
                            "proof": (
                                f"strict data-propagating shape {shape}; reduced axes "
                                f"{normalized} contain exactly one element and keepdims=1"
                            ),
                        }
            if lead is None:
                source_shape = metadata.get(node.input[0], {}).get("shape")
                output_shape = metadata.get(node.output[0], {}).get("shape")
                direct = direct_unsigned_cast_identity(node, producers, metadata)
                if direct is not None and source_shape == output_shape:
                    lead = direct
        if lead is not None:
            lead.update({
                "task": task,
                "node_index": node_index,
                "op": node.op_type,
                "source": node.input[0],
                "output": node.output[0],
                "source_consumers": len(consumers[node.input[0]]),
            })
            bypasses.append(lead)
            row["exact_identity"] = lead
        rows.append(row)

    return {
        "task": task,
        "authority_sha256": sha256(data),
        "strict_data_prop": strict_error is None,
        "strict_data_prop_error": strict_error,
        "op_counts": dict(op_counts),
        "target_nodes": rows,
        "bypasses": bypasses,
        "deduplications": deduplications,
    }


def apply_rewrites(
    model: onnx.ModelProto,
    bypasses: list[dict[str, Any]],
    deduplications: list[dict[str, Any]],
) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    replacements = {item["output"]: item["source"] for item in bypasses}
    replacements.update({item["removed"]: item["kept"] for item in deduplications})
    remove_outputs = set(replacements)
    for node in result.graph.node:
        for index, name in enumerate(node.input):
            while name in replacements:
                name = replacements[name]
            node.input[index] = name
    keep_nodes = [
        node for node in result.graph.node
        if not any(name in remove_outputs for name in node.output)
    ]
    del result.graph.node[:]
    result.graph.node.extend(keep_nodes)
    used = {name for node in result.graph.node for name in node.input if name}
    used.update(value.name for value in result.graph.input)
    used.update(value.name for value in result.graph.output)
    keep_initializers = [item for item in result.graph.initializer if item.name in used]
    del result.graph.initializer[:]
    result.graph.initializer.extend(keep_initializers)
    live = used | {name for node in result.graph.node for name in node.output if name}
    keep_vi = [value for value in result.graph.value_info if value.name in live]
    del result.graph.value_info[:]
    result.graph.value_info.extend(keep_vi)
    onnx.checker.check_model(result, full_check=True)
    onnx.shape_inference.infer_shapes(result, strict_mode=True, data_prop=True)
    return result


def profile(model: onnx.ModelProto, prefix: str) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as directory:
        path = Path(directory) / "model.onnx"
        onnx.save(model, path)
        return cost_of(str(path))


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    models = []
    total_counts: Counter[str] = Counter()
    candidates = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            data = archive.read(f"task{task:03d}.onnx")
            result = scan_model(task, data)
            models.append(result)
            total_counts.update(result["op_counts"])
            if not result["bypasses"] and not result["deduplications"]:
                continue
            source = onnx.load_model_from_string(data)
            try:
                candidate = apply_rewrites(
                    source, result["bypasses"], result["deduplications"]
                )
                base_profile = profile(source, f"di241_base_{task:03d}_")
                cand_profile = profile(candidate, f"di241_cand_{task:03d}_")
            except Exception as exc:  # noqa: BLE001
                candidates.append({
                    "task": task,
                    "authority_sha256": result["authority_sha256"],
                    "bypasses": result["bypasses"],
                    "deduplications": result["deduplications"],
                    "build_error": f"{type(exc).__name__}: {exc}",
                    "strict_lower": False,
                })
                continue
            row = {
                "task": task,
                "authority_sha256": result["authority_sha256"],
                "bypasses": result["bypasses"],
                "deduplications": result["deduplications"],
                "baseline": {"memory": base_profile[0], "params": base_profile[1], "cost": base_profile[2]},
                "candidate": {"memory": cand_profile[0], "params": cand_profile[1], "cost": cand_profile[2]},
                "strict_lower": cand_profile[2] < base_profile[2],
            }
            if row["strict_lower"]:
                path = CANDIDATES / f"task{task:03d}_domain_identity.onnx"
                onnx.save(candidate, path)
                row.update({
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path.read_bytes()),
                    "gain": base_profile[2] - cand_profile[2],
                    "score_gain": math.log(base_profile[2] / cand_profile[2]),
                })
            candidates.append(row)
    candidates.sort(key=lambda item: (-int(item.get("gain", 0)), item["task"]))
    runtime_audit_path = HERE / "audit.json"
    runtime_audits = {}
    if runtime_audit_path.exists():
        runtime_audits = {
            int(item["task"]): item
            for item in json.loads(runtime_audit_path.read_text()).get("tasks", [])
        }
    for row in candidates:
        audit = runtime_audits.get(int(row["task"]))
        if audit is not None:
            row["runtime_gate"] = {
                "accepted": bool(audit.get("accepted")),
                "known_case_count": audit.get("known_case_count"),
                "candidate_shape_truthful": audit.get("candidate_shapes", {}).get("truthful"),
                "fresh_seeds_run": len(audit.get("fresh", [])),
                "evidence": "audit.json",
            }
            row["admissible"] = bool(row.get("strict_lower")) and bool(audit.get("accepted"))
        elif row["task"] == 366:
            row["runtime_gate"] = {
                "accepted": False,
                "candidate_shape_truthful": False,
                "reason": "declared singleton source [1] executes as [15] during official profiling",
            }
            row["admissible"] = False
        else:
            row["admissible"] = False
    output = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": sha256(AUTHORITY.read_bytes()),
        "models_scanned": 400,
        "target_op_counts": dict(sorted(total_counts.items())),
        "mechanical_census": {
            "pow_scalar_exponent_0_1_2": 0,
            "neutral_scalar_arithmetic_connections": 5,
            "unsigned_clip_redundant_min0_inputs": 5,
            "fully_dtype_determined_comparisons": 0,
            "selu_gamma_exactly_one": 0,
            "shape_dtype_byte_exact_duplicate_scalar_groups": 1,
            "duplicate_scalar_group": {"task": 264, "kept": "s1", "removed": "axes1"},
            "strict_rejected_unsigned_cast_leakyrelu_identities": 3,
            "strict_rejected_activation_task": 243,
        },
        "strict_data_prop_pass_models": sum(item["strict_data_prop"] for item in models),
        "strict_data_prop_fail_models": sum(not item["strict_data_prop"] for item in models),
        "exact_static_lead_count": sum(
            len(item["bypasses"]) + len(item["deduplications"]) for item in models
        ),
        "strict_lower_candidate_count": sum(bool(item.get("strict_lower")) for item in candidates),
        "admissible_candidate_count": sum(bool(item.get("admissible")) for item in candidates),
        "candidates": candidates,
        "models": models,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "models_scanned", "strict_data_prop_pass_models", "strict_data_prop_fail_models",
        "exact_static_lead_count", "strict_lower_candidate_count",
    )}, indent=2))


if __name__ == "__main__":
    main()
