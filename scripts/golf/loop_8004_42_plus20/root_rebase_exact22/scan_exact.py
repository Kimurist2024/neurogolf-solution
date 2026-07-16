#!/usr/bin/env python3
"""Find narrow exact rewrites in the remaining 8005.16 changed members.

This is only a candidate builder.  It never mutates the baseline ZIP and never
promotes a candidate without the separate dual-ORT known/fresh safety gate.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8005.16.zip"
CANDIDATES = HERE / "candidates"
TASKS = (133, 145, 182, 187, 201, 204, 216, 233, 255, 319, 330, 349, 361, 367, 370)

EXACT_MODULE = HERE.parent / "agent_exact_wave2" / "scan_and_build.py"
spec = importlib.util.spec_from_file_location("exact_wave2", EXACT_MODULE)
assert spec is not None and spec.loader is not None
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def emit(task: int, label: str, model: onnx.ModelProto, base_cost: int,
         detail: dict[str, object]) -> dict[str, object]:
    audit, _ = exact.structural_audit(model)
    record: dict[str, object] = {
        "task": task,
        "label": label,
        "detail": detail,
        "base_cost": base_cost,
        "structural": audit,
        "status": "structural_reject",
    }
    if not audit.get("pass"):
        return record
    cost = int(audit["cost"])
    record["candidate_cost"] = cost
    record["cost_reduction"] = base_cost - cost
    if cost >= base_cost:
        record["status"] = "no_cost_reduction"
        return record
    path = CANDIDATES / f"task{task:03d}_{label}.onnx"
    onnx.save(model, path)
    record.update({
        "status": "emitted",
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path.read_bytes()),
    })
    return record


def main() -> int:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "baseline": BASE_ZIP.name,
        "baseline_sha256": sha256(BASE_ZIP.read_bytes()),
        "tasks": list(TASKS),
        "records": [],
    }
    records: list[dict[str, object]] = report["records"]  # type: ignore[assignment]
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            base_audit, inferred = exact.structural_audit(model)
            if not base_audit.get("pass") or inferred is None:
                records.append({
                    "task": task,
                    "label": "baseline",
                    "status": "baseline_structural_reject",
                    "structural": base_audit,
                })
                continue
            base_cost = int(base_audit["cost"])
            graph_outputs = {value.name for value in model.graph.output}
            consumers: dict[str, int] = {}
            for node in model.graph.node:
                for value in node.input:
                    if value:
                        consumers[value] = consumers.get(value, 0) + 1
            type_map = {
                value.name: value
                for value in list(inferred.graph.input)
                + list(inferred.graph.value_info)
                + list(inferred.graph.output)
            }

            # Byte-identical initializer aliases.
            canonical: dict[bytes, str] = {}
            replacements: dict[str, str] = {}
            for initializer in model.graph.initializer:
                key = exact.tensor_key(initializer)
                if key in canonical:
                    replacements[initializer.name] = canonical[key]
                else:
                    canonical[key] = initializer.name
            if replacements:
                candidate = copy.deepcopy(model)
                for node in candidate.graph.node:
                    for index, value in enumerate(node.input):
                        if value in replacements:
                            node.input[index] = replacements[value]
                kept = [item for item in candidate.graph.initializer if item.name not in replacements]
                del candidate.graph.initializer[:]
                candidate.graph.initializer.extend(kept)
                exact.remove_value_info(candidate, set(replacements))
                records.append(emit(task, "initializer_alias", candidate, base_cost,
                                    {"replacements": replacements}))

            # Output-unreachable nodes/initializers.
            live, needed = exact.backwards_slice(model)
            dead = [index for index in range(len(model.graph.node)) if index not in set(live)]
            unused = [item.name for item in model.graph.initializer if item.name not in needed]
            if dead or unused:
                candidate = copy.deepcopy(model)
                removed_outputs = {
                    output for index in dead for output in model.graph.node[index].output if output
                }
                kept_nodes = [candidate.graph.node[index] for index in live]
                kept_initializers = [item for item in candidate.graph.initializer if item.name in needed]
                del candidate.graph.node[:]
                candidate.graph.node.extend(kept_nodes)
                del candidate.graph.initializer[:]
                candidate.graph.initializer.extend(kept_initializers)
                exact.remove_value_info(candidate, removed_outputs | set(unused))
                records.append(emit(task, "dead_code", candidate, base_cost,
                                    {"dead_nodes": dead, "unused_initializers": unused}))

            # Internal Identity removal, one isolated candidate per node.
            for index, node in enumerate(model.graph.node):
                if (node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1
                        or not node.input[0] or not node.output[0]
                        or node.output[0] in graph_outputs):
                    continue
                candidate = copy.deepcopy(model)
                exact.replace_uses(candidate, node.output[0], node.input[0])
                del candidate.graph.node[index]
                removed = exact.prune_unused_initializers(candidate)
                exact.remove_value_info(candidate, {node.output[0], *removed})
                records.append(emit(task, f"identity_{index:03d}", candidate, base_cost,
                                    {"index": index, "input": node.input[0],
                                     "output": node.output[0], "removed_initializers": removed}))

            # Provably no-op Cast and literal same-shape Reshape.
            initializers = {item.name: item for item in model.graph.initializer}
            for index, node in enumerate(model.graph.node):
                if (len(node.input) < 1 or len(node.output) != 1 or not node.input[0]
                        or not node.output[0] or node.output[0] in graph_outputs):
                    continue
                reason: str | None = None
                source = type_map.get(node.input[0])
                target = type_map.get(node.output[0])
                if node.op_type == "Cast" and source is not None and target is not None:
                    to_type = next((attr.i for attr in node.attribute if attr.name == "to"), None)
                    if (to_type is not None
                            and source.type.tensor_type.elem_type
                            == target.type.tensor_type.elem_type == to_type):
                        reason = "same_dtype_cast"
                elif (node.op_type == "Reshape" and len(node.input) == 2
                      and source is not None and target is not None):
                    shape_tensor = initializers.get(node.input[1])
                    if shape_tensor is not None:
                        literal = [int(value) for value in
                                   exact.numpy_helper.to_array(shape_tensor).reshape(-1)]
                        if literal == exact.dims(source) == exact.dims(target):
                            reason = "literal_same_shape_reshape"
                if reason is None:
                    continue
                candidate = copy.deepcopy(model)
                exact.replace_uses(candidate, node.output[0], node.input[0])
                del candidate.graph.node[index]
                removed = exact.prune_unused_initializers(candidate)
                exact.remove_value_info(candidate, {node.output[0], *removed})
                records.append(emit(task, f"noop_{index:03d}", candidate, base_cost,
                                    {"index": index, "op": node.op_type, "reason": reason,
                                     "removed_initializers": removed}))

            # Duplicate deterministic producers with identical payload and inputs.
            producer: dict[bytes, int] = {}
            for index, node in enumerate(model.graph.node):
                key = exact.node_key(node)
                previous = producer.get(key)
                if previous is None:
                    producer[key] = index
                    continue
                first = model.graph.node[previous]
                if (node.op_type in exact.NONDETERMINISTIC
                        or len(first.output) != len(node.output)
                        or any(output in graph_outputs for output in node.output if output)):
                    continue
                pairs = [(old, new) for old, new in zip(node.output, first.output) if old and new]
                if len(pairs) != len([output for output in node.output if output]):
                    continue
                candidate = copy.deepcopy(model)
                for old, new in pairs:
                    exact.replace_uses(candidate, old, new)
                del candidate.graph.node[index]
                removed = exact.prune_unused_initializers(candidate)
                exact.remove_value_info(candidate,
                                        {old for old, _ in pairs} | set(removed))
                records.append(emit(task, f"duplicate_{index:03d}", candidate, base_cost,
                                    {"index": index, "canonical": previous,
                                     "op": node.op_type, "replacements": dict(pairs),
                                     "removed_initializers": removed}))

            # Unused optional secondary outputs may be omitted exactly.
            for node_index, node in enumerate(model.graph.node):
                for output_index in range(1, len(node.output)):
                    output = node.output[output_index]
                    if not output or output in graph_outputs or consumers.get(output, 0):
                        continue
                    candidate = copy.deepcopy(model)
                    candidate.graph.node[node_index].output[output_index] = ""
                    exact.remove_value_info(candidate, {output})
                    records.append(emit(task, f"optional_{node_index:03d}_{output_index}",
                                        candidate, base_cost,
                                        {"node_index": node_index,
                                         "output_index": output_index,
                                         "op": node.op_type, "output": output}))

    report["summary"] = {
        "tasks_scanned": len(TASKS),
        "records": len(records),
        "emitted": sum(row.get("status") == "emitted" for row in records),
        "structural_rejects": sum(row.get("status") in {
            "structural_reject", "baseline_structural_reject"
        } for row in records),
    }
    (HERE / "scan_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
