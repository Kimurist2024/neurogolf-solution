#!/usr/bin/env python3
"""Eliminate exact Concat(Split(x)) round trips across all 400 tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

AUTHORITY = REPO / "submission_base_8009.46.zip"
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates"


def attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"splitconcat255_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


def prune(model: onnx.ModelProto) -> tuple[list[str], list[str]]:
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    graph_outputs = {item.name for item in model.graph.output}
    removed_nodes: list[str] = []
    changed = True
    while changed:
        changed = False
        keep = []
        for node in model.graph.node:
            if node.output and all(uses[name] == 0 and name not in graph_outputs for name in node.output if name):
                removed_nodes.append(node.name or node.op_type)
                for name in node.input:
                    if name:
                        uses[name] -= 1
                changed = True
            else:
                keep.append(node)
        if changed:
            del model.graph.node[:]
            model.graph.node.extend(keep)
    graph_inputs = {item.name for item in model.graph.input}
    removed_inits = [
        item.name for item in model.graph.initializer
        if uses[item.name] == 0 and item.name not in graph_inputs
    ]
    keep_inits = [item for item in model.graph.initializer if item.name not in removed_inits]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_inits)
    return removed_nodes, removed_inits


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            producer = {
                output: (index, node)
                for index, node in enumerate(model.graph.node)
                for output in node.output if output
            }
            baseline = None
            for index, node in enumerate(model.graph.node):
                if node.op_type != "Concat" or not node.input:
                    continue
                source = None
                description = None
                if len(node.input) == 1:
                    source = node.input[0]
                    description = "single-input Concat"
                else:
                    parents = [producer.get(name) for name in node.input]
                    if any(parent is None for parent in parents):
                        continue
                    parent_nodes = [parent[1] for parent in parents if parent is not None]
                    split = parent_nodes[0]
                    if split.op_type != "Split" or any(parent is not split for parent in parent_nodes):
                        continue
                    if list(node.input) != list(split.output):
                        continue
                    if attr_int(node, "axis", 0) != attr_int(split, "axis", 0):
                        continue
                    source = split.input[0]
                    description = "Concat(all Split outputs in order)"
                candidate = copy.deepcopy(model)
                original = candidate.graph.node[index]
                original.CopyFrom(helper.make_node(
                    "Identity", [source], list(original.output), name=original.name
                ))
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
        "identities": len(rows),
        "strict_lower": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("tasks", "identities", "strict_lower")}, indent=2))


if __name__ == "__main__":
    main()
