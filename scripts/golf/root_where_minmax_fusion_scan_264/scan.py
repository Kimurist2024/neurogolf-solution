#!/usr/bin/env python3
"""Fuse exact comparison/Where selectors into Min, Max, or Identity."""

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


def profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"whereminmax264_{task:03d}_") as wd:
        path = Path(wd) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": memory, "params": params, "cost": cost}


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


def replacement(
    where: onnx.NodeProto,
    condition: onnx.NodeProto,
) -> tuple[str, list[str], str] | None:
    if len(condition.input) != 2 or len(where.input) != 3:
        return None
    left, right = condition.input
    yes, no = where.input[1], where.input[2]
    if condition.op_type in {"Greater", "GreaterOrEqual"}:
        if (yes, no) == (left, right):
            return "Max", [left, right], "Where(a>b,a,b)->Max"
        if (yes, no) == (right, left):
            return "Min", [left, right], "Where(a>b,b,a)->Min"
    if condition.op_type in {"Less", "LessOrEqual"}:
        if (yes, no) == (left, right):
            return "Min", [left, right], "Where(a<b,a,b)->Min"
        if (yes, no) == (right, left):
            return "Max", [left, right], "Where(a<b,b,a)->Max"
    if condition.op_type == "Equal":
        if (yes, no) == (right, left):
            return "Identity", [left], "Where(a==b,b,a)->a"
        if (yes, no) == (left, right):
            return "Identity", [right], "Where(a==b,a,b)->b"
    return None


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    census = {"Where": 0, "comparison_where": 0, "patterns": 0}
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
                if node.op_type != "Where":
                    continue
                census["Where"] += 1
                parent = producer.get(node.input[0])
                if parent is None or parent[1].op_type not in {
                    "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal"
                }:
                    continue
                census["comparison_where"] += 1
                found = replacement(node, parent[1])
                if found is None:
                    continue
                census["patterns"] += 1
                op_type, inputs, description = found
                candidate = copy.deepcopy(model)
                original = candidate.graph.node[index]
                original.CopyFrom(helper.make_node(
                    op_type, inputs, list(original.output), name=original.name
                ))
                removed_nodes, removed_inits = prune(candidate)
                record: dict = {
                    "task": task,
                    "node_index": index,
                    "condition_index": parent[0],
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
