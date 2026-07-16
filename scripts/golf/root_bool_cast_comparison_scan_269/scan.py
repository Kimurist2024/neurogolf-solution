#!/usr/bin/env python3
"""Eliminate numeric Cast(bool) immediately consumed by a scalar comparison."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
CANDIDATES = HERE / "candidates"
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

COMPARISONS = {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"boolcmp269_{task:03d}_{label}_") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def truth(op: str, left: float | int, right: float | int) -> bool:
    if op == "Equal":
        return bool(left == right)
    if op == "Greater":
        return bool(left > right)
    if op == "GreaterOrEqual":
        return bool(left >= right)
    if op == "Less":
        return bool(left < right)
    if op == "LessOrEqual":
        return bool(left <= right)
    raise AssertionError(op)


def prune(model: onnx.ModelProto) -> tuple[list[str], list[str]]:
    outputs = {value.name for value in model.graph.output}
    removed_nodes: list[str] = []
    while True:
        uses = Counter(name for node in model.graph.node for name in node.input if name)
        doomed = {
            index for index, node in enumerate(model.graph.node)
            if node.output and all(uses[name] == 0 and name not in outputs for name in node.output if name)
        }
        if not doomed:
            break
        keep = []
        for index, node in enumerate(model.graph.node):
            if index in doomed:
                removed_nodes.append(node.name or node.op_type)
            else:
                keep.append(node)
        del model.graph.node[:]
        model.graph.node.extend(keep)
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    graph_inputs = {value.name for value in model.graph.input}
    removed_initializers = [
        item.name for item in model.graph.initializer
        if uses[item.name] == 0 and item.name not in graph_inputs
    ]
    keep_initializers = [item for item in model.graph.initializer if item.name not in removed_initializers]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep_initializers)
    return removed_nodes, removed_initializers


def main() -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    census = Counter()
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=False, data_prop=True)
            types = {
                value.name: value.type.tensor_type.elem_type
                for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
                if value.type.HasField("tensor_type")
            }
            producers = {
                output: (index, node)
                for index, node in enumerate(model.graph.node)
                for output in node.output if output
            }
            constants = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
            baseline = None
            for index, node in enumerate(model.graph.node):
                if node.op_type not in COMPARISONS or len(node.input) != 2:
                    continue
                census["comparisons"] += 1
                for cast_position in (0, 1):
                    parent = producers.get(node.input[cast_position])
                    constant = constants.get(node.input[1 - cast_position])
                    if parent is None or parent[1].op_type not in {"Cast", "CastLike"}:
                        continue
                    cast = parent[1]
                    if not cast.input or types.get(cast.input[0]) != TensorProto.BOOL:
                        continue
                    if constant is None or constant.size != 1 or constant.dtype.kind not in "fiu":
                        continue
                    value = constant.reshape(-1)[0]
                    if constant.dtype.kind == "f" and not np.isfinite(value):
                        continue
                    census["bool_cast_scalar_comparisons"] += 1
                    outcomes = []
                    for bit in (False, True):
                        numeric = int(bit)
                        left, right = (numeric, value) if cast_position == 0 else (value, numeric)
                        outcomes.append(truth(node.op_type, left, right))
                    if outcomes == [False, True]:
                        replacement_op = "Identity"
                    elif outcomes == [True, False]:
                        replacement_op = "Not"
                    else:
                        continue
                    census["identity_or_not"] += 1
                    candidate = copy.deepcopy(model)
                    original = candidate.graph.node[index]
                    original.CopyFrom(helper.make_node(
                        replacement_op,
                        [cast.input[0]],
                        list(original.output),
                        name=original.name,
                    ))
                    removed_nodes, removed_initializers = prune(candidate)
                    record = {
                        "task": task,
                        "node_index": index,
                        "cast_index": parent[0],
                        "comparison": node.op_type,
                        "cast_position": cast_position,
                        "constant": repr(value.item() if hasattr(value, "item") else value),
                        "replacement": replacement_op,
                        "removed_nodes": removed_nodes,
                        "removed_initializers": removed_initializers,
                    }
                    try:
                        onnx.checker.check_model(candidate, full_check=True)
                        onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                        if baseline is None:
                            baseline = profile(model, task, "authority")
                        current = profile(candidate, task, f"node{index}")
                        record["baseline"] = baseline
                        record["candidate"] = current
                        record["strict_lower"] = current["cost"] < baseline["cost"]
                        if record["strict_lower"]:
                            path = CANDIDATES / f"task{task:03d}_{index:04d}.onnx"
                            onnx.save(candidate, path)
                            record["path"] = str(path.relative_to(ROOT))
                            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(record)
    result = {
        "authority": str(AUTHORITY),
        "tasks": len(members),
        "census": dict(census),
        "strict_lower": sum(bool(row.get("strict_lower")) for row in rows),
        "rows": rows,
    }
    (HERE / "scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"census": dict(census), "strict_lower": result["strict_lower"], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
