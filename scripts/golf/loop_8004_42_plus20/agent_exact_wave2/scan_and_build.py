#!/usr/bin/env python3
"""Scan the 8004.50 payload for narrow, semantics-preserving ONNX rewrites.

This lane intentionally does not merge a ZIP.  It inventories all 400 models and
emits isolated candidates for mechanical rewrites only:

* byte-identical initializer aliases;
* output-unreachable nodes/initializers (except known allocator failures);
* internal Identity and provably no-op Cast/Reshape nodes;
* duplicate deterministic producers;
* unused optional secondary outputs;
* truthful value_info reductions derived after clearing annotations.

Every emitted candidate must pass the full ONNX checker, strict shape inference
with data propagation, static-positive-shape checks, banned-op checks and the
Conv-family bias-length check.  Runtime/gold/fresh validation is deliberately a
separate gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE / "base"
CANDIDATE_DIR = HERE / "candidates"
REPORT = HERE / "static_scan.json"

BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
NONDETERMINISTIC = {
    "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
    "Multinomial", "Bernoulli", "Dropout",
}

# Removing these dead nodes was already tried on the 8003.40 payload and made
# every known case fail at runtime because the incumbent depends on allocator /
# liveness behavior.  Repeating that experiment would add no information.
PAST_DEAD_RUNTIME_FAILURES = {39, 89, 111, 122, 183}

# Prior exact-lane rejections.  We inventory these opportunities but do not
# rebuild/re-admit them: task048 missed the 95% fresh gate, task233 is a repeated
# private-zero dust gain, and task333 changes a giant floating contraction.
PAST_EXACT_REJECTIONS = {48, 233, 333}

# Catalogued private-zero / unsound-incumbent tasks.  This lane is not allowed
# to manufacture a new candidate in their lineage, even from a local rewrite.
PRIVATE_RISK = {
    9, 15, 35, 44, 48, 66, 72, 77, 86, 90, 96, 101, 102, 133, 134,
    138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187, 192, 196,
    202, 205, 209, 216, 219, 222, 233, 246, 255, 277, 285, 286, 302,
    325, 346, 361, 365, 366, 372, 377, 379, 393, 396,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    shape = value.type.tensor_type.shape
    result: list[int] = []
    for dim in shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def tensor_key(tensor: onnx.TensorProto) -> bytes:
    clone = onnx.TensorProto()
    clone.CopyFrom(tensor)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def node_key(node: onnx.NodeProto) -> bytes:
    clone = onnx.NodeProto()
    clone.CopyFrom(node)
    clone.name = ""
    del clone.output[:]
    return clone.SerializeToString(deterministic=True)


def conv_bias_ub(model: onnx.ModelProto, inferred: onnx.ModelProto) -> list[dict[str, Any]]:
    values = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    initializers = {item.name: item for item in inferred.graph.initializer}
    failures: list[dict[str, Any]] = []
    for node in inferred.graph.node:
        bias_index = 8 if node.op_type == "QLinearConv" else (
            2 if node.op_type in {"Conv", "ConvTranspose"} else None
        )
        if bias_index is None or len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        bias = initializers.get(node.input[bias_index])
        output = values.get(node.output[0]) if node.output else None
        output_shape = dims(output) if output is not None else None
        if bias is None or output_shape is None or len(output_shape) < 2:
            failures.append({"node": node.name, "op": node.op_type, "reason": "unknown"})
            continue
        bias_len = int(np.prod(bias.dims, dtype=np.int64)) if bias.dims else 1
        if bias_len != output_shape[1]:
            failures.append({
                "node": node.name,
                "op": node.op_type,
                "bias": bias_len,
                "output_channels": output_shape[1],
            })
    return failures


def static_cost(model: onnx.ModelProto, inferred: onnx.ModelProto) -> dict[str, int]:
    infos = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    excluded = {item.name for item in inferred.graph.input}
    excluded.update(item.name for item in inferred.graph.output)
    excluded.update(item.name for item in inferred.graph.initializer)
    memory = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for output_name in node.output:
            if not output_name or output_name in excluded or output_name in seen:
                continue
            seen.add(output_name)
            value = infos.get(output_name)
            shape = dims(value) if value is not None else None
            if shape is None:
                raise ValueError(f"non-static node output: {output_name}")
            dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            memory += math.prod(shape) * np.dtype(dtype).itemsize
    params = 0
    for item in inferred.graph.initializer:
        params += math.prod(item.dims) if item.dims else 1
    for sparse in inferred.graph.sparse_initializer:
        params += math.prod(sparse.values.dims) if sparse.values.dims else 1
    for node in inferred.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                params += math.prod(attr.t.dims) if attr.t.dims else 1
            elif attr.name == "sparse_value":
                params += math.prod(attr.sparse_tensor.values.dims) if attr.sparse_tensor.values.dims else 1
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return {"memory": int(memory), "params": int(params), "cost": int(memory + params)}


def structural_audit(model: onnx.ModelProto) -> tuple[dict[str, Any], onnx.ModelProto | None]:
    audit: dict[str, Any] = {
        "checker": False,
        "strict_shape_data_prop": False,
        "static_positive": False,
        "banned_ops": [],
        "nested_graphs": [],
        "conv_bias_ub": [],
        "errors": [],
        "cost": None,
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        audit["checker"] = True
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        audit["strict_shape_data_prop"] = True
    except Exception as exc:
        audit["errors"].append(f"checker_or_inference:{type(exc).__name__}:{exc}")
        return audit, None

    if model.functions:
        audit["errors"].append("model_functions")
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED or "SEQUENCE" in upper:
            audit["banned_ops"].append(node.op_type)
        for attr in node.attribute:
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                audit["nested_graphs"].append(f"{node.name}/{attr.name}")

    static = True
    for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if dims(item) is None:
            static = False
            audit["errors"].append(f"nonstatic:{item.name}")
    audit["static_positive"] = static
    audit["conv_bias_ub"] = conv_bias_ub(model, inferred)
    try:
        audit.update(static_cost(model, inferred))
    except Exception as exc:
        audit["errors"].append(f"cost:{type(exc).__name__}:{exc}")
    audit["pass"] = bool(
        audit["checker"]
        and audit["strict_shape_data_prop"]
        and audit["static_positive"]
        and not audit["banned_ops"]
        and not audit["nested_graphs"]
        and not audit["conv_bias_ub"]
        and not audit["errors"]
        and audit.get("cost") is not None
    )
    return audit, inferred


def remove_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def replace_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def prune_unused_initializers(model: onnx.ModelProto) -> list[str]:
    used = {value for node in model.graph.node for value in node.input if value}
    used.update(output.name for output in model.graph.output)
    removed = [item.name for item in model.graph.initializer if item.name not in used]
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def save_candidate(task: int, label: str, model: onnx.ModelProto, base_audit: dict[str, Any],
                   detail: dict[str, Any]) -> dict[str, Any]:
    audit, _ = structural_audit(model)
    record: dict[str, Any] = {
        "task": task,
        "label": label,
        "detail": detail,
        "base_static": {key: base_audit.get(key) for key in ("memory", "params", "cost")},
        "candidate_static": {key: audit.get(key) for key in ("memory", "params", "cost")},
        "structural": audit,
        "path": None,
        "static_gain": None,
    }
    if not audit.get("pass"):
        record["status"] = "structural_reject"
        return record
    candidate_cost = int(audit["cost"])
    base_cost = int(base_audit["cost"])
    record["static_gain"] = base_cost - candidate_cost
    if candidate_cost >= base_cost:
        record["status"] = "no_cost_reduction"
        return record
    path = CANDIDATE_DIR / f"task{task:03d}_{label}.onnx"
    onnx.save(model, path)
    record["path"] = str(path.relative_to(HERE.parents[3]))
    record["sha256"] = sha256_bytes(path.read_bytes())
    record["status"] = "emitted"
    return record


def backwards_slice(model: onnx.ModelProto) -> tuple[list[int], set[str]]:
    needed = {output.name for output in model.graph.output}
    live: list[int] = []
    for index in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[index]
        if any(output and output in needed for output in node.output):
            live.append(index)
            needed.update(value for value in node.input if value)
    live.reverse()
    return live, needed


def main() -> int:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "baseline": "submission_base_8004.50.zip",
        "baseline_sha256": "63cb4c2abf794bb3cc0ceb531db907625c82638656e7d1ab29865d39b42a6cac",
        "tasks_scanned": 0,
        "baseline_structural_failures": [],
        "opportunities": {
            "initializer_alias": [],
            "dead_code": [],
            "noop_node": [],
            "duplicate_producer": [],
            "dead_optional_output": [],
            "annotation_reduction": [],
        },
        "candidates": [],
        "past_failure_exclusions": {
            "dead_runtime": sorted(PAST_DEAD_RUNTIME_FAILURES),
            "exact_rejections": sorted(PAST_EXACT_REJECTIONS),
            "private_risk": sorted(PRIVATE_RISK),
        },
    }

    for task in range(1, 401):
        source = BASE_DIR / f"task{task:03d}.onnx"
        model = onnx.load(source, load_external_data=False)
        report["tasks_scanned"] += 1
        base_audit, inferred = structural_audit(model)
        if not base_audit.get("pass") or inferred is None:
            report["baseline_structural_failures"].append({
                "task": task,
                "audit": base_audit,
            })
            # Candidate admission requires strict data-propagating inference.
            continue

        output_names = {item.name for item in model.graph.output}
        consumers: dict[str, int] = {}
        for node in model.graph.node:
            for value in node.input:
                if value:
                    consumers[value] = consumers.get(value, 0) + 1
        type_map = {
            item.name: item
            for item in list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        }

        # 1. Byte-identical initializer aliases.
        canonical: dict[bytes, str] = {}
        replacements: dict[str, str] = {}
        for initializer in model.graph.initializer:
            key = tensor_key(initializer)
            if key in canonical:
                replacements[initializer.name] = canonical[key]
            else:
                canonical[key] = initializer.name
        if replacements:
            opportunity = {"task": task, "replacements": replacements}
            report["opportunities"]["initializer_alias"].append(opportunity)
            if task not in PRIVATE_RISK and task not in PAST_EXACT_REJECTIONS:
                candidate = copy.deepcopy(model)
                for node in candidate.graph.node:
                    for index, value in enumerate(node.input):
                        if value in replacements:
                            node.input[index] = replacements[value]
                kept = [item for item in candidate.graph.initializer if item.name not in replacements]
                del candidate.graph.initializer[:]
                candidate.graph.initializer.extend(kept)
                report["candidates"].append(save_candidate(
                    task, "initializer_alias", candidate, base_audit, opportunity
                ))

        # 2. Output-unreachable nodes and initializers.
        live, needed = backwards_slice(model)
        live_set = set(live)
        dead_nodes = [index for index in range(len(model.graph.node)) if index not in live_set]
        unused_initializers = [item.name for item in model.graph.initializer if item.name not in needed]
        if dead_nodes or unused_initializers:
            opportunity = {
                "task": task,
                "dead_nodes": [
                    {"index": index, "op": model.graph.node[index].op_type,
                     "outputs": list(model.graph.node[index].output)}
                    for index in dead_nodes
                ],
                "unused_initializers": unused_initializers,
                "past_runtime_failure": task in PAST_DEAD_RUNTIME_FAILURES,
            }
            report["opportunities"]["dead_code"].append(opportunity)
            if task not in PAST_DEAD_RUNTIME_FAILURES and task not in PRIVATE_RISK:
                candidate = copy.deepcopy(model)
                removed_outputs = {
                    output for index in dead_nodes for output in model.graph.node[index].output if output
                }
                kept_nodes = [candidate.graph.node[index] for index in live]
                kept_initializers = [item for item in candidate.graph.initializer if item.name in needed]
                del candidate.graph.node[:]
                candidate.graph.node.extend(kept_nodes)
                del candidate.graph.initializer[:]
                candidate.graph.initializer.extend(kept_initializers)
                remove_value_info(candidate, removed_outputs | set(unused_initializers))
                report["candidates"].append(save_candidate(
                    task, "dead_code", candidate, base_audit, opportunity
                ))

        # 3. Internal no-op Identity/Cast/Reshape nodes, one candidate each.
        for index, node in enumerate(model.graph.node):
            if len(node.input) < 1 or len(node.output) != 1 or not node.input[0] or not node.output[0]:
                continue
            output = node.output[0]
            if output in output_names:
                continue
            reason: str | None = None
            if node.op_type == "Identity":
                reason = "Identity"
            elif node.op_type == "Cast":
                input_value = type_map.get(node.input[0])
                output_value = type_map.get(output)
                to_attr = next((attr.i for attr in node.attribute if attr.name == "to"), None)
                if (
                    input_value is not None and output_value is not None and to_attr is not None
                    and input_value.type.tensor_type.elem_type == output_value.type.tensor_type.elem_type == to_attr
                ):
                    reason = "same_dtype_Cast"
            elif node.op_type == "Reshape":
                input_value = type_map.get(node.input[0])
                output_value = type_map.get(output)
                if (
                    input_value is not None and output_value is not None
                    and dims(input_value) == dims(output_value)
                    and input_value.type.tensor_type.elem_type == output_value.type.tensor_type.elem_type
                ):
                    # Require the target shape to be a literal exactly equal to
                    # the source's positive static shape; this avoids treating a
                    # declared-shape cloak as a no-op.
                    init = {item.name: item for item in model.graph.initializer}
                    target = init.get(node.input[1]) if len(node.input) > 1 else None
                    if target is not None:
                        try:
                            target_shape = [int(value) for value in numpy_helper.to_array(target).reshape(-1)]
                        except Exception:
                            target_shape = []
                        if target_shape == dims(input_value):
                            reason = "literal_same_shape_Reshape"
            if reason is None:
                continue
            opportunity = {
                "task": task, "index": index, "op": node.op_type,
                "input": node.input[0], "output": output, "reason": reason,
            }
            report["opportunities"]["noop_node"].append(opportunity)
            if task in PRIVATE_RISK:
                continue
            candidate = copy.deepcopy(model)
            replace_uses(candidate, output, node.input[0])
            del candidate.graph.node[index]
            removed_initializers = prune_unused_initializers(candidate)
            remove_value_info(candidate, {output, *removed_initializers})
            detail = {**opportunity, "removed_initializers": removed_initializers}
            report["candidates"].append(save_candidate(
                task, f"noop_{index:03d}", candidate, base_audit, detail
            ))

        # 4. Duplicate deterministic producers.
        producer: dict[bytes, int] = {}
        for index, node in enumerate(model.graph.node):
            key = node_key(node)
            first_index = producer.get(key)
            if first_index is None:
                producer[key] = index
                continue
            first = model.graph.node[first_index]
            if node.op_type in NONDETERMINISTIC or len(first.output) != len(node.output):
                continue
            if any(output in output_names for output in node.output if output):
                continue
            pairs = [(old, new) for old, new in zip(node.output, first.output) if old and new]
            if len(pairs) != len([output for output in node.output if output]):
                continue
            opportunity = {
                "task": task, "duplicate_index": index, "canonical_index": first_index,
                "op": node.op_type, "replacements": dict(pairs),
            }
            report["opportunities"]["duplicate_producer"].append(opportunity)
            if task in PRIVATE_RISK:
                continue
            candidate = copy.deepcopy(model)
            for old, new in pairs:
                replace_uses(candidate, old, new)
            del candidate.graph.node[index]
            removed_initializers = prune_unused_initializers(candidate)
            remove_value_info(candidate, {old for old, _ in pairs} | set(removed_initializers))
            detail = {**opportunity, "removed_initializers": removed_initializers}
            report["candidates"].append(save_candidate(
                task, f"duplicate_{index:03d}", candidate, base_audit, detail
            ))

        # 5. Named but unused secondary node outputs.  We only omit output
        # positions >=1; the validator and scorer require a non-empty first.
        for node_index, node in enumerate(model.graph.node):
            if len(node.output) <= 1:
                continue
            for output_index in range(1, len(node.output)):
                output = node.output[output_index]
                if not output or output in output_names or consumers.get(output, 0):
                    continue
                opportunity = {
                    "task": task, "node_index": node_index, "output_index": output_index,
                    "op": node.op_type, "output": output,
                }
                report["opportunities"]["dead_optional_output"].append(opportunity)
                if task in PRIVATE_RISK:
                    continue
                candidate = copy.deepcopy(model)
                candidate.graph.node[node_index].output[output_index] = ""
                remove_value_info(candidate, {output})
                report["candidates"].append(save_candidate(
                    task, f"optional_{node_index:03d}_{output_index}", candidate, base_audit, opportunity
                ))

        # 6. Truthful annotation reductions.  Clear value_info and let ONNX
        # reconstruct it from computational payload only, then retain smaller
        # positive static shapes for existing annotations.
        clean = copy.deepcopy(model)
        del clean.graph.value_info[:]
        try:
            clean_inferred = shape_inference.infer_shapes(clean, strict_mode=True, data_prop=True)
        except Exception:
            clean_inferred = None
        if clean_inferred is not None:
            clean_map = {item.name: item for item in clean_inferred.graph.value_info}
            reductions: list[dict[str, Any]] = []
            for value in model.graph.value_info:
                truthful = clean_map.get(value.name)
                old_shape = dims(value)
                new_shape = dims(truthful) if truthful is not None else None
                if old_shape is None or new_shape is None or len(old_shape) != len(new_shape):
                    continue
                if math.prod(new_shape) < math.prod(old_shape):
                    reductions.append({"name": value.name, "from": old_shape, "to": new_shape})
            if reductions:
                opportunity = {"task": task, "reductions": reductions}
                report["opportunities"]["annotation_reduction"].append(opportunity)
                if task not in PRIVATE_RISK:
                    candidate = copy.deepcopy(model)
                    cand_map = {item.name: item for item in candidate.graph.value_info}
                    for reduction in reductions:
                        truthful = clean_map[reduction["name"]]
                        cand_map[reduction["name"]].CopyFrom(truthful)
                    report["candidates"].append(save_candidate(
                        task, "annotation_reduction", candidate, base_audit, opportunity
                    ))

    report["summary"] = {
        "tasks_scanned": report["tasks_scanned"],
        "baseline_structural_failures": len(report["baseline_structural_failures"]),
        "opportunity_counts": {
            key: len(value) for key, value in report["opportunities"].items()
        },
        "candidate_records": len(report["candidates"]),
        "emitted": sum(row.get("status") == "emitted" for row in report["candidates"]),
        "structural_rejects": sum(row.get("status") == "structural_reject" for row in report["candidates"]),
        "no_cost_reduction": sum(row.get("status") == "no_cost_reduction" for row in report["candidates"]),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
