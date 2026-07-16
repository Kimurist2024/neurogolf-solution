#!/usr/bin/env python3
"""Audit dead nodes/outputs and build only lineage-safe dead-code candidates."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "lane_a38"))
from scripts.golf.rank_dir import cost_of  # noqa: E402
import scan_build_exact_cse as safety  # noqa: E402


AUTHORITY = ROOT / "submission_base_8000.46.zip"
AUTHORITY_SHA256 = "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534"
NAMED_TARGETS = {
    39: "keep_bg_equal",
    89: "keep_red_big",
    122: "d_keep",
    183: "hold_u8",
}


def opset_for(model: onnx.ModelProto, domain: str) -> int | None:
    normalized = domain or "ai.onnx"
    for item in model.opset_import:
        item_domain = item.domain or "ai.onnx"
        if item_domain == normalized:
            return int(item.version)
    return None


def schema_output_options(model: onnx.ModelProto, node: onnx.NodeProto) -> list[str]:
    version = opset_for(model, node.domain)
    if version is None:
        return ["unknown"] * len(node.output)
    try:
        schema = onnx.defs.get_schema(node.op_type, version, node.domain)
    except Exception:  # noqa: BLE001
        return ["unknown"] * len(node.output)
    options = []
    for index in range(len(node.output)):
        if index >= len(schema.outputs):
            options.append("variadic_extension")
            continue
        options.append(str(schema.outputs[index].option).split(".")[-1])
    return options


def live_node_indices(model: onnx.ModelProto) -> set[int]:
    needed = {value.name for value in model.graph.output}
    live: set[int] = set()
    for index in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[index]
        if any(output and output in needed for output in node.output):
            live.add(index)
            needed.update(name for name in node.input if name)
    return live


def prune(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    candidate = copy.deepcopy(model)
    live = live_node_indices(candidate)
    dead_nodes = [
        {
            "index": index,
            "op": node.op_type,
            "name": node.name,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        for index, node in enumerate(candidate.graph.node)
        if index not in live
    ]
    kept_nodes = [node for index, node in enumerate(candidate.graph.node) if index in live]
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept_nodes)
    used = (
        {value.name for value in candidate.graph.input}
        | {value.name for value in candidate.graph.output}
        | {name for node in candidate.graph.node for name in node.input if name}
    )
    dead_initializers = [item for item in candidate.graph.initializer if item.name not in used]
    kept_initializers = [item for item in candidate.graph.initializer if item.name in used]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept_initializers)
    retained_names = used | {name for node in candidate.graph.node for name in node.output if name}
    kept_info = [value for value in candidate.graph.value_info if value.name in retained_names]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(kept_info)
    onnx.checker.check_model(candidate, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    if not all(safety.static_shape(value) for value in values):
        raise RuntimeError("candidate has non-static shape")
    return candidate, {
        "dead_nodes": dead_nodes,
        "dead_initializers": [
            {
                "name": item.name,
                "elements": int(np.prod(item.dims, dtype=np.int64)) if item.dims else 1,
            }
            for item in dead_initializers
        ],
    }


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"a39_{task:03d}_") as directory:
        path = Path(directory) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        return tuple(int(value) for value in cost_of(str(path)))


def main() -> None:
    if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest() != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP hash mismatch")
    candidates_dir = HERE / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    task_rows: list[dict[str, object]] = []
    multi_output_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            payload = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model(io.BytesIO(payload))
            live = live_node_indices(model)
            exclusions = safety.source_exclusions(model)
            consumed = {name for node in model.graph.node for name in node.input if name}
            consumed |= {value.name for value in model.graph.output}
            dead = [
                {
                    "index": index,
                    "op": node.op_type,
                    "name": node.name,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                }
                for index, node in enumerate(model.graph.node)
                if index not in live
            ]
            partial_outputs = []
            for index, node in enumerate(model.graph.node):
                if len(node.output) <= 1:
                    continue
                unused = [
                    {"index": output_index, "name": name}
                    for output_index, name in enumerate(node.output)
                    if name and name not in consumed
                ]
                if not unused:
                    continue
                row = {
                    "task": task,
                    "node_index": index,
                    "op": node.op_type,
                    "name": node.name,
                    "outputs": list(node.output),
                    "unused_outputs": unused,
                    "schema_output_options": schema_output_options(model, node),
                    "source_exclusions": exclusions,
                }
                partial_outputs.append(row)
                multi_output_rows.append(row)
            named = NAMED_TARGETS.get(task)
            named_match = [row for row in dead if named and named in row["outputs"]]
            if dead or partial_outputs or named:
                task_rows.append(
                    {
                        "task": task,
                        "source_sha256": hashlib.sha256(payload).hexdigest(),
                        "source_exclusions": exclusions,
                        "dead_nodes": dead,
                        "partial_multi_outputs": partial_outputs,
                        "named_target": named,
                        "named_target_dead_match": named_match,
                        "op_histogram": dict(Counter(node.op_type for node in model.graph.node)),
                    }
                )
            if not dead or exclusions:
                continue
            try:
                candidate, changes = prune(model)
                baseline_memory, baseline_parameters, baseline_cost = measure(model, task)
                memory, parameters, cost = measure(candidate, task)
                row = {
                    "task": task,
                    "source_sha256": hashlib.sha256(payload).hexdigest(),
                    "baseline_memory": baseline_memory,
                    "baseline_parameters": baseline_parameters,
                    "baseline_cost": baseline_cost,
                    "candidate_memory": memory,
                    "candidate_parameters": parameters,
                    "candidate_cost": cost,
                    **changes,
                }
                if cost < baseline_cost:
                    path = candidates_dir / f"task{task:03d}.onnx"
                    onnx.save(candidate, path)
                    row["candidate"] = str(path.relative_to(ROOT))
                    row["candidate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    row["projected_gain"] = float(np.log(baseline_cost / cost))
                    candidate_rows.append(row)
                else:
                    row["reason"] = "no_actual_cost_reduction"
                    errors.append(row)
            except Exception as exc:  # noqa: BLE001
                errors.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
    candidate_rows.sort(key=lambda row: -float(row["projected_gain"]))
    manifest = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks_scanned": 400,
        "named_targets": NAMED_TARGETS,
        "tasks_with_dead_nodes_or_multi_outputs": task_rows,
        "partial_multi_output_count": len(multi_output_rows),
        "partial_multi_outputs": multi_output_rows,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "errors": errors,
        "authority_zip_modified": False,
    }
    (HERE / "scan_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "tasks_with_findings": len(task_rows),
        "dead_node_tasks": [row["task"] for row in task_rows if row["dead_nodes"]],
        "partial_multi_output_count": len(multi_output_rows),
        "candidate_count": len(candidate_rows),
        "candidate_tasks": [row["task"] for row in candidate_rows],
        "errors": len(errors),
    }, indent=2))


if __name__ == "__main__":
    main()
