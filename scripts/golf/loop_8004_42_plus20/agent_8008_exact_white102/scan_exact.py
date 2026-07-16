#!/usr/bin/env python3
"""Mechanical all-input-exact golf scan over the 37 new 8008.14 members."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import defs, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD_ZIP = ROOT / "others/71403/lb_verified_8006.61/submission.zip"
BASE_ZIP = ROOT / "submission_base_8008.14.zip"
BASE_SHA256 = "50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6"
KINDS = ("cleanup", "dedupe", "noops", "cse", "optional", "fold", "absorb", "combined", "normalize", "normalized_combined")

sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from harvest import actual_screen, known_score  # noqa: E402

HELPER_PATH = ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py"
SPEC = importlib.util.spec_from_file_location("lane102_helpers", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load audit helpers")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def remove_unused_initializers(model: onnx.ModelProto) -> list[str]:
    live = {name for node in model.graph.node for name in node.input if name}
    live.update(value.name for value in model.graph.input)
    live.update(value.name for value in model.graph.output)
    removed = [item.name for item in model.graph.initializer if item.name not in live]
    if removed:
        removed_set = set(removed)
        keep = [item for item in model.graph.initializer if item.name not in removed_set]
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
    return removed


def remove_dead_nodes(model: onnx.ModelProto) -> list[dict[str, Any]]:
    actions = []
    while True:
        nodes = list(model.graph.node)
        used = {name for node in model.graph.node for name in node.input if name}
        used.update(value.name for value in model.graph.output)
        dead_indices = {
            index for index, node in enumerate(nodes)
            if all(not name or name not in used for name in node.output)
        }
        dead = [nodes[index] for index in sorted(dead_indices)]
        if not dead:
            break
        actions.extend({"op": node.op_type, "outputs": list(node.output)} for node in dead)
        keep = [node for index, node in enumerate(nodes) if index not in dead_indices]
        del model.graph.node[:]
        model.graph.node.extend(keep)
    remove_unused_initializers(model)
    return actions


def remove_dead_value_info(model: onnx.ModelProto) -> list[str]:
    live = {value.name for value in model.graph.input}
    live.update(value.name for value in model.graph.output)
    live.update(name for node in model.graph.node for name in (*node.input, *node.output) if name)
    removed = [value.name for value in model.graph.value_info if value.name not in live]
    if removed:
        removed_set = set(removed)
        keep = [value for value in model.graph.value_info if value.name not in removed_set]
        del model.graph.value_info[:]
        model.graph.value_info.extend(keep)
    return removed


def clear_value_info(model: onnx.ModelProto) -> list[str]:
    """Remove advisory intermediate shapes; graph execution is unchanged."""
    removed = [value.name for value in model.graph.value_info]
    del model.graph.value_info[:]
    return removed


def initializer_key(item: onnx.TensorProto) -> tuple[Any, ...]:
    clone = copy.deepcopy(item)
    clone.name = ""
    return int(item.data_type), tuple(item.dims), clone.SerializeToString()


def dedupe_initializers(model: onnx.ModelProto) -> list[dict[str, str]]:
    protected = {value.name for value in model.graph.input}
    protected.update(value.name for value in model.graph.output)
    first = {}
    replacements = {}
    for item in model.graph.initializer:
        if item.name in protected:
            continue
        key = initializer_key(item)
        if key in first:
            replacements[item.name] = first[key]
        else:
            first[key] = item.name
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
    if replacements:
        keep = [item for item in model.graph.initializer if item.name not in replacements]
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
    return [{"removed": old, "reused": new} for old, new in sorted(replacements.items())]


def replace_uses(model: onnx.ModelProto, target: str, source: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == target:
                node.input[index] = source


def tensor_shape(value: onnx.ValueInfoProto | None) -> tuple[int, ...] | None:
    if value is None or not value.type.HasField("tensor_type"):
        return None
    dims = value.type.tensor_type.shape.dim
    if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in dims):
        return None
    return tuple(int(dim.dim_value) for dim in dims)


def bypass_noops(model: onnx.ModelProto) -> list[dict[str, Any]]:
    try:
        inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    except Exception:
        inferred = copy.deepcopy(model)
    values = {value.name: value for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)}
    arrays = {}
    for item in model.graph.initializer:
        try:
            arrays[item.name] = numpy_helper.to_array(item)
        except Exception:
            pass
    graph_outputs = {value.name for value in model.graph.output}
    producer = {name: node for node in model.graph.node for name in node.output if name}
    actions = []
    nodes = list(model.graph.node)
    remove_indices = set()
    for node_index, node in enumerate(nodes):
        if len(node.output) != 1 or not node.output[0] or node.output[0] in graph_outputs:
            continue
        source = reason = None
        if node.op_type == "Identity" and len(node.input) == 1:
            source, reason = node.input[0], "identity"
        elif node.op_type in {"Concat", "Sum", "Max", "Min"} and len(node.input) == 1:
            source, reason = node.input[0], "single_input_variadic"
        elif node.op_type in {"Add", "Mul", "Or", "And", "Xor"} and len(node.input) == 2:
            neutral = {"Add": 0, "Mul": 1, "Or": False, "And": True, "Xor": False}[node.op_type]
            for index in (0, 1):
                array = arrays.get(node.input[index])
                if array is not None and array.size and np.all(array == neutral):
                    source, reason = node.input[1 - index], f"{node.op_type.lower()}_neutral"
                    break
        elif node.op_type in {"Sub", "Div"} and len(node.input) == 2:
            neutral = 0 if node.op_type == "Sub" else 1
            array = arrays.get(node.input[1])
            if array is not None and array.size and np.all(array == neutral):
                source, reason = node.input[0], f"{node.op_type.lower()}_right_neutral"
        elif node.op_type == "Where" and len(node.input) == 3:
            condition = arrays.get(node.input[0])
            if condition is not None and condition.size and np.all(condition):
                source, reason = node.input[1], "where_constant_true"
            elif condition is not None and condition.size and not np.any(condition):
                source, reason = node.input[2], "where_constant_false"
        elif node.op_type == "Transpose" and len(node.input) == 1:
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            perm = attrs.get("perm")
            if perm is not None and list(perm) == list(range(len(perm))):
                source, reason = node.input[0], "identity_transpose"
        elif node.op_type == "Cast" and len(node.input) == 1:
            before, after = values.get(node.input[0]), values.get(node.output[0])
            if before is not None and after is not None and before.type.tensor_type.elem_type == after.type.tensor_type.elem_type:
                source, reason = node.input[0], "same_dtype_cast"
        elif node.op_type == "Reshape" and node.input:
            if tensor_shape(values.get(node.input[0])) == tensor_shape(values.get(node.output[0])):
                source, reason = node.input[0], "same_shape_reshape"
        elif node.op_type == "Pad" and len(node.input) >= 2:
            pads = arrays.get(node.input[1])
            if pads is not None and np.all(pads == 0):
                source, reason = node.input[0], "zero_pad"
        elif node.op_type in {"Not", "Neg"} and len(node.input) == 1:
            first = producer.get(node.input[0])
            if first is not None and first.op_type == node.op_type and len(first.input) == 1:
                source, reason = first.input[0], f"double_{node.op_type.lower()}"
        if source:
            replace_uses(model, node.output[0], source)
            remove_indices.add(node_index)
            actions.append({"op": node.op_type, "output": node.output[0], "source": source, "proof": reason})
    if remove_indices:
        keep = [node for index, node in enumerate(nodes) if index not in remove_indices]
        del model.graph.node[:]
        model.graph.node.extend(keep)
        remove_dead_nodes(model)
        remove_unused_initializers(model)
    return actions


PURE_CSE = {
    "Shape", "Cast", "CastLike", "CenterCropPad", "Add", "Sub", "Mul", "Div",
    "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Xor",
    "Not", "Neg", "Concat", "Reshape", "Transpose", "Slice", "Gather", "GatherElements",
    "Unsqueeze", "Squeeze", "ConstantOfShape", "Where", "Max", "Min", "Mod", "BitwiseAnd",
    "BitwiseOr", "BitwiseXor", "BitShift", "Log", "Sqrt", "Abs", "Ceil", "Floor", "Round",
}


def node_signature(node: onnx.NodeProto) -> tuple[Any, ...]:
    attrs = tuple(sorted((attr.name, attr.SerializeToString()) for attr in node.attribute))
    return node.domain, node.op_type, tuple(node.input), attrs, len(node.output)


def common_subexpressions(model: onnx.ModelProto) -> list[dict[str, Any]]:
    graph_outputs = {value.name for value in model.graph.output}
    first = {}
    nodes = list(model.graph.node)
    remove_indices = set()
    actions = []
    for node_index, node in enumerate(nodes):
        if node.op_type not in PURE_CSE or any(name in graph_outputs for name in node.output if name):
            continue
        signature = node_signature(node)
        prior = first.get(signature)
        if prior is None:
            first[signature] = node
            continue
        if len(prior.output) != len(node.output) or any(not name for name in prior.output):
            continue
        for target, source in zip(node.output, prior.output):
            if target:
                replace_uses(model, target, source)
        remove_indices.add(node_index)
        actions.append({"op": node.op_type, "removed_outputs": list(node.output), "reused_outputs": list(prior.output), "proof": "identical_node_signature"})
    if remove_indices:
        keep = [node for index, node in enumerate(nodes) if index not in remove_indices]
        del model.graph.node[:]
        model.graph.node.extend(keep)
        remove_dead_nodes(model)
    return actions


def remove_optional_outputs(model: onnx.ModelProto) -> list[dict[str, Any]]:
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(value.name for value in model.graph.output)
    opsets = {item.domain: int(item.version) for item in model.opset_import}
    actions = []
    for node in model.graph.node:
        domain = node.domain or ""
        version = opsets.get(domain, opsets.get("", 18))
        try:
            schema = defs.get_schema(node.op_type, version, domain)
        except Exception:
            continue
        while node.output and node.output[-1] and node.output[-1] not in used:
            index = len(node.output) - 1
            if index >= len(schema.outputs) or schema.outputs[index].option != defs.OpSchema.FormalParameterOption.Optional:
                break
            removed = node.output[-1]
            del node.output[-1]
            actions.append({"op": node.op_type, "removed_output": removed, "proof": "schema_optional_and_unused"})
    return actions


def fold_array(node: onnx.NodeProto, arrays: dict[str, np.ndarray]) -> np.ndarray | None:
    if len(node.output) != 1 or any(name not in arrays for name in node.input if name):
        return None
    xs = [arrays[name] for name in node.input if name]
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    try:
        if node.op_type == "Identity": return xs[0]
        if node.op_type == "Cast": return xs[0].astype(helper.tensor_dtype_to_np_dtype(int(attrs["to"])))
        if node.op_type == "Reshape": return np.reshape(xs[0], tuple(int(v) for v in xs[1].flat))
        if node.op_type == "Transpose": return np.transpose(xs[0], axes=attrs.get("perm"))
        if node.op_type == "Concat": return np.concatenate(xs, axis=int(attrs.get("axis", 0)))
        if node.op_type == "Add": return np.add(xs[0], xs[1])
        if node.op_type == "Sub": return np.subtract(xs[0], xs[1])
        if node.op_type == "Mul": return np.multiply(xs[0], xs[1])
        if node.op_type == "Div": return np.divide(xs[0], xs[1]).astype(xs[0].dtype)
        if node.op_type == "Neg": return np.negative(xs[0])
        if node.op_type == "Not": return np.logical_not(xs[0])
        if node.op_type == "Equal": return np.equal(xs[0], xs[1])
        if node.op_type == "Less": return np.less(xs[0], xs[1])
        if node.op_type == "Greater": return np.greater(xs[0], xs[1])
        if node.op_type == "And": return np.logical_and(xs[0], xs[1])
        if node.op_type == "Or": return np.logical_or(xs[0], xs[1])
        if node.op_type == "Xor": return np.logical_xor(xs[0], xs[1])
        if node.op_type == "Shape":
            start = int(attrs.get("start", 0)); end = int(attrs.get("end", xs[0].ndim))
            return np.asarray(xs[0].shape[start:end], dtype=np.int64)
        if node.op_type == "Squeeze":
            axes = tuple(int(v) for v in (xs[1].flat if len(xs) > 1 else attrs.get("axes", [])))
            return np.squeeze(xs[0], axis=axes or None)
        if node.op_type == "Unsqueeze":
            result = xs[0]
            axes = sorted(int(v) for v in (xs[1].flat if len(xs) > 1 else attrs.get("axes", [])))
            for axis in axes: result = np.expand_dims(result, axis)
            return result
    except Exception:
        return None
    return None


def constant_fold(model: onnx.ModelProto) -> list[dict[str, Any]]:
    actions = []
    while True:
        arrays = {}
        for item in model.graph.initializer:
            try: arrays[item.name] = numpy_helper.to_array(item)
            except Exception: pass
        folded = None
        nodes = list(model.graph.node)
        for node_index, node in enumerate(nodes):
            array = fold_array(node, arrays)
            if array is not None:
                folded = (node_index, node, np.ascontiguousarray(array))
                break
        if folded is None:
            break
        node_index, node, array = folded
        output = node.output[0]
        keep_init = [item for item in model.graph.initializer if item.name != output]
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep_init)
        model.graph.initializer.append(numpy_helper.from_array(array, output))
        keep_nodes = [item for index, item in enumerate(nodes) if index != node_index]
        del model.graph.node[:]
        model.graph.node.extend(keep_nodes)
        actions.append({"op": node.op_type, "output": output, "shape": list(array.shape), "dtype": str(array.dtype), "proof": "all_inputs_constant_numpy_equivalent"})
        remove_dead_nodes(model)
    remove_unused_initializers(model)
    return actions


def factor_absorb(model: onnx.ModelProto) -> list[dict[str, Any]]:
    nodes = list(model.graph.node)
    arrays = {}
    for item in model.graph.initializer:
        try: arrays[item.name] = numpy_helper.to_array(item)
        except Exception: pass
    consumers = defaultdict(list)
    for node in nodes:
        for name in node.input:
            if name: consumers[name].append(node)
    producers = {
        name: (index, node)
        for index, node in enumerate(nodes)
        for name in node.output if name
    }
    graph_outputs = {value.name for value in model.graph.output}
    actions = []
    remove_indices = set()
    for node in nodes:
        if node.op_type not in {"Add", "Mul"} or len(node.input) != 2:
            continue
        current_const_index = next((i for i, name in enumerate(node.input) if name in arrays), None)
        if current_const_index is None:
            continue
        link_index = 1 - current_const_index
        producer_entry = producers.get(node.input[link_index])
        if producer_entry is None:
            continue
        producer_index, producer = producer_entry
        if producer.op_type != node.op_type or len(producer.input) != 2:
            continue
        if producer.output[0] in graph_outputs or len(consumers[producer.output[0]]) != 1:
            continue
        prior_const_index = next((i for i, name in enumerate(producer.input) if name in arrays), None)
        if prior_const_index is None:
            continue
        try:
            left = arrays[producer.input[prior_const_index]]
            right = arrays[node.input[current_const_index]]
            combined = np.add(left, right) if node.op_type == "Add" else np.multiply(left, right)
            combined = np.ascontiguousarray(combined)
        except Exception:
            continue
        new_name = f"exact_absorb_{node.op_type.lower()}_{len(actions)}"
        model.graph.initializer.append(numpy_helper.from_array(combined, new_name))
        node.input[link_index] = producer.input[1 - prior_const_index]
        node.input[current_const_index] = new_name
        remove_indices.add(producer_index)
        actions.append({
            "op": node.op_type,
            "removed_intermediate": producer.output[0],
            "combined_initializers": [producer.input[prior_const_index], node.input[current_const_index]],
            "new_initializer": new_name,
            "proof": "associativity_in_exact_real_or_integer_semantics",
        })
    if remove_indices:
        keep = [node for index, node in enumerate(nodes) if index not in remove_indices]
        del model.graph.node[:]
        model.graph.node.extend(keep)
        remove_dead_nodes(model)
        remove_unused_initializers(model)
    return actions


def transform(base: onnx.ModelProto, kind: str) -> tuple[onnx.ModelProto, dict[str, Any]]:
    model = copy.deepcopy(base)
    detail: dict[str, Any] = {}
    combined = kind in {"combined", "normalized_combined"}
    if kind in {"cleanup", "combined", "normalized_combined"}:
        detail["dead_nodes"] = remove_dead_nodes(model)
        detail["unused_initializers"] = remove_unused_initializers(model)
        detail["dead_value_info"] = remove_dead_value_info(model)
    if kind in {"dedupe", "combined", "normalized_combined"}: detail["deduplicated_initializers"] = dedupe_initializers(model)
    if kind in {"optional", "combined", "normalized_combined"}: detail["removed_optional_outputs"] = remove_optional_outputs(model)
    if kind in {"noops", "combined", "normalized_combined"}: detail["bypassed_noops"] = bypass_noops(model)
    if kind in {"cse", "combined", "normalized_combined"}: detail["common_subexpressions"] = common_subexpressions(model)
    if kind in {"fold", "combined", "normalized_combined"}: detail["constant_folds"] = constant_fold(model)
    if kind in {"absorb", "combined", "normalized_combined"}: detail["factor_absorptions"] = factor_absorb(model)
    if combined:
        # A second pass catches opportunities exposed by the first pass.
        detail["second_noops"] = bypass_noops(model)
        detail["second_cse"] = common_subexpressions(model)
        detail["final_dead_nodes"] = remove_dead_nodes(model)
        detail["final_unused_initializers"] = remove_unused_initializers(model)
        detail["final_dead_value_info"] = remove_dead_value_info(model)
    if kind in {"normalize", "normalized_combined"}:
        detail["cleared_value_info"] = clear_value_info(model)
    detail["semantic_action_count"] = sum(
        len(value) for key, value in detail.items()
        if isinstance(value, list) and key not in {"dead_value_info", "final_dead_value_info", "cleared_value_info"}
    )
    detail["metadata_action_count"] = sum(
        len(value) for key, value in detail.items()
        if isinstance(value, list) and key in {"dead_value_info", "final_dead_value_info", "cleared_value_info"}
    )
    return model, detail


def load_costs(tasks: list[int]) -> dict[int, int]:
    costs = {}
    lines = (ROOT / "all_scores.csv").read_text().splitlines()[1:]
    for line in lines:
        cells = line.split(",")
        if len(cells) >= 4 and cells[1].startswith("task"):
            task = int(cells[1][4:])
            if task in tasks: costs[task] = int(cells[3])
    if set(costs) != set(tasks):
        raise RuntimeError("all_scores does not cover all 37 targets")
    return costs


def main() -> int:
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    (HERE / "audit").mkdir(exist_ok=True)
    if sha256(BASE_ZIP.read_bytes()) != BASE_SHA256:
        raise RuntimeError("8008.14 authority hash mismatch")
    with zipfile.ZipFile(OLD_ZIP) as old, zipfile.ZipFile(BASE_ZIP) as new:
        tasks = [task for task in range(1, 401) if old.read(f"task{task:03d}.onnx") != new.read(f"task{task:03d}.onnx")]
        if len(tasks) != 37:
            raise RuntimeError(f"expected 37 exact-white changes, got {len(tasks)}")
        payloads = {task: new.read(f"task{task:03d}.onnx") for task in tasks}
    costs = load_costs(tasks)

    rows = []
    observations = []
    seen = set()
    for task in tasks:
        base_data = payloads[task]
        base = onnx.load_model_from_string(base_data)
        observations.append({
            "task": task,
            "sha256": sha256(base_data),
            "cost": costs[task],
            "nodes": len(base.graph.node),
            "initializers": len(base.graph.initializer),
            "value_info": len(base.graph.value_info),
            "initializer_dtype_note": "parameter cost counts elements, not bytes; dtype-only narrowing cannot reduce score",
        })
        for kind in KINDS:
            model, actions = transform(base, kind)
            if not actions["semantic_action_count"] and not actions["metadata_action_count"]:
                continue
            data = model.SerializeToString()
            candidate_sha = sha256(data)
            if candidate_sha == sha256(base_data) or (task, candidate_sha) in seen:
                continue
            seen.add((task, candidate_sha))
            path = HERE / "candidates" / f"task{task:03d}_{kind}_{candidate_sha[:12]}.onnx"
            path.write_bytes(data)
            row = {
                "task": task,
                "kind": kind,
                "path": rel(path),
                "sha256": candidate_sha,
                "authority_sha256": sha256(base_data),
                "authority_cost": costs[task],
                "actions": actions,
                "exact_structure_proof": [
                    item.get("proof", key)
                    for key, value in actions.items() if isinstance(value, list)
                    for item in value if isinstance(item, dict)
                ],
            }
            structural = AUDIT.structural_audit(data)
            row["structural_audit"] = structural
            if not structural["pass"]:
                row["stage"] = "REJECT_STRUCTURE_SCHEMA_UB"
                rows.append(row)
                continue
            actual = actual_screen(data, task)
            row["actual_screen_cost"] = actual
            if actual is None or actual >= costs[task]:
                row["stage"] = "REJECT_NOT_STRICTLY_LOWER"
                rows.append(row)
                continue
            try:
                profile = known_score(data, task, False, f"lane102_{task}_{kind}_{candidate_sha[:8]}")
            except Exception as exc:  # noqa: BLE001
                row["official_profile_error"] = f"{type(exc).__name__}: {exc}"
                row["stage"] = "REJECT_OFFICIAL_PROFILE"
                rows.append(row)
                continue
            row["official_profile"] = profile
            if not profile or not profile.get("correct") or int(profile["cost"]) >= costs[task]:
                row["stage"] = "REJECT_OFFICIAL_NOT_CORRECT_LOWER"
                rows.append(row)
                continue
            row["candidate_cost"] = int(profile["cost"])
            row["gain"] = math.log(costs[task] / int(profile["cost"]))
            quad = AUDIT.known_four(task, data)
            row["known_four"] = quad
            if not AUDIT.known_complete(quad):
                row["stage"] = "REJECT_KNOWN4_OR_RUNTIME"
                rows.append(row)
                continue
            try:
                trace = AUDIT.runtime_shape_trace(task, onnx.load_model_from_string(data))
                row["runtime_shape_trace"] = trace
            except Exception as exc:  # noqa: BLE001
                row["runtime_shape_trace_error"] = f"{type(exc).__name__}: {exc}"
                row["stage"] = "REJECT_SHAPE_TRACE"
                rows.append(row)
                continue
            if trace["declared_actual_mismatches"]:
                row["stage"] = "REJECT_SHAPE_CLOAK"
                rows.append(row)
                continue
            row["stage"] = "EXACT_CANDIDATE"
            rows.append(row)
        print(f"SCAN task{task:03d} candidates={sum(1 for row in rows if row['task'] == task)}", flush=True)

    exact = sorted([row for row in rows if row["stage"] == "EXACT_CANDIDATE"], key=lambda row: (-row["gain"], row["task"], row["sha256"]))
    report = {
        "authority_zip": "submission_base_8008.14.zip",
        "authority_zip_sha256": BASE_SHA256,
        "targets": tasks,
        "target_count": len(tasks),
        "candidate_count": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "exact_candidate_count": len(exact),
        "exact_candidates": exact,
        "observations": observations,
        "rows": rows,
    }
    (HERE / "audit/mechanical_scan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "target_count": len(tasks),
        "candidate_count": len(rows),
        "stage_counts": report["stage_counts"],
        "exact_candidate_count": len(exact),
        "exact_tasks": dict(Counter(row["task"] for row in exact)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
