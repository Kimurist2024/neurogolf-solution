#!/usr/bin/env python3
"""Rank archived ONNX candidates against the immutable 8003.40 baseline.

This is a read-only archive scan.  It deliberately uses a conservative
structural filter before any expensive known-example/runtime validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import onnx


ROOT = Path(__file__).resolve().parents[4]
BASE_DIR = ROOT / "scripts/golf/loop_8003_40/base_models"
SCORES = ROOT / "all_scores.csv"
ARCHIVE_ROOTS = (ROOT / "others",)
OUT = Path(__file__).with_name("archive_static_scan.json")

TASK_RE = re.compile(r"task[_-]?(\d{1,3})", re.IGNORECASE)
PRIVATE_OR_UNSOUND = {
    9, 15, 35, 44, 48, 66, 72, 77, 86, 90, 96, 101, 102, 133,
    134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 202, 205, 209, 216, 219, 222, 233, 246, 255, 277,
    285, 286, 302, 325, 346, 361, 365, 366, 372, 377, 379, 393,
    396,
}
ALREADY_ADOPTED = {13, 105, 132, 153, 158, 344, 349, 358, 379, 398}
ALREADY_SCREENED = {23, 131, 143, 168, 200, 201, 251, 328}
EXCLUDED_TASKS = PRIVATE_OR_UNSOUND | ALREADY_ADOPTED | ALREADY_SCREENED
EXCLUDED_OPS = {
    "Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress",
    "TfIdfVectorizer",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def task_from_path(path: Path) -> int | None:
    match = TASK_RE.search(path.name)
    if not match:
        return None
    task = int(match.group(1))
    return task if 1 <= task <= 400 else None


def tensor_elements(tensor: onnx.TensorProto) -> int:
    return math.prod(tensor.dims) if tensor.dims else 1


def params(model: onnx.ModelProto) -> int:
    total = sum(tensor_elements(x) for x in model.graph.initializer)
    total += sum(tensor_elements(x.values) for x in model.graph.sparse_initializer)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                total += tensor_elements(attr.t)
            elif attr.name == "sparse_value":
                total += tensor_elements(attr.sparse_tensor.values)
            elif attr.name == "value_floats":
                total += len(attr.floats)
            elif attr.name == "value_ints":
                total += len(attr.ints)
            elif attr.name == "value_strings":
                total += len(attr.strings)
    return int(total)


def static_memory(model: onnx.ModelProto) -> int:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    infos = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    init_names = {item.name for item in inferred.graph.initializer}
    init_names.update(item.name for item in inferred.graph.sparse_initializer)
    graph_io = {item.name for item in inferred.graph.input}
    graph_io.update(item.name for item in inferred.graph.output)
    total = 0
    seen: set[str] = set()
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in seen or name in graph_io or name in init_names:
                continue
            seen.add(name)
            info = infos.get(name)
            if info is None or not info.type.HasField("tensor_type"):
                raise ValueError(f"missing tensor info: {name}")
            tt = info.type.tensor_type
            if not tt.HasField("shape"):
                raise ValueError(f"missing shape: {name}")
            elements = 1
            for dim in tt.shape.dim:
                if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                    raise ValueError(f"non-static shape: {name}")
                elements *= int(dim.dim_value)
            dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
            total += elements * np.dtype(dtype).itemsize
    return int(total)


def structural_audit(model: onnx.ModelProto, baseline: onnx.ModelProto) -> tuple[bool, list[str], dict[str, object]]:
    reasons: list[str] = []
    ops = [node.op_type for node in model.graph.node]
    if model.functions:
        reasons.append("functions")
    if any(op in EXCLUDED_OPS or "Sequence" in op for op in ops):
        reasons.append("excluded_or_lookup_op")
    nested = any(
        attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        for node in model.graph.node
        for attr in node.attribute
    )
    if nested:
        reasons.append("nested_graph")
    candidate_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    baseline_einsum = max(
        (len(node.input) for node in baseline.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    if candidate_einsum >= 20 and candidate_einsum > baseline_einsum:
        reasons.append("giant_einsum_expansion")
    initializer_elements = {
        init.name: tensor_elements(init) for init in model.graph.initializer
    }
    suspicious_gather_tables = []
    for node in model.graph.node:
        if node.op_type not in {"Gather", "GatherElements", "GatherND"} or not node.input:
            continue
        elements = initializer_elements.get(node.input[0], 0)
        if elements >= 256:
            suspicious_gather_tables.append({"name": node.input[0], "elements": elements})
    # Do not auto-reject Gather: many exact spatial/permutation nets use it.
    # Surface it for mandatory manual lineage review instead.
    details = {
        "ops": sorted(set(ops)),
        "candidate_max_einsum_inputs": candidate_einsum,
        "baseline_max_einsum_inputs": baseline_einsum,
        "suspicious_gather_tables": suspicious_gather_tables,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
    }
    return not reasons, reasons, details


def main() -> None:
    costs: dict[int, int] = {}
    with SCORES.open(newline="") as f:
        for row in csv.DictReader(f):
            costs[int(row["task"].removeprefix("task"))] = int(row["cost"])

    baselines = {
        task: onnx.load(BASE_DIR / f"task{task:03d}.onnx", load_external_data=False)
        for task in range(1, 401)
        if task not in EXCLUDED_TASKS
    }
    baseline_sha = {
        task: sha256(BASE_DIR / f"task{task:03d}.onnx") for task in baselines
    }

    paths_seen = 0
    size_prefiltered = 0
    unique_seen: set[tuple[int, str]] = set()
    records: list[dict[str, object]] = []
    rejects: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejects[reason] = rejects.get(reason, 0) + 1

    for archive_root in ARCHIVE_ROOTS:
        for dirpath, _, filenames in os.walk(archive_root):
            for filename in filenames:
                if not filename.lower().endswith(".onnx"):
                    continue
                path = Path(dirpath) / filename
                task = task_from_path(path)
                if task is None or task in EXCLUDED_TASKS:
                    continue
                paths_seen += 1
                # Useful golf nets are compact.  This generous cap still admits
                # large parameter encodings while avoiding megabyte lookup blobs.
                cap = min(1_450_000, max(100_000, costs[task] * 32 + 20_000))
                if path.stat().st_size > cap:
                    reject("serialized_size_prefilter")
                    continue
                size_prefiltered += 1
                try:
                    digest = sha256(path)
                except OSError:
                    reject("read_error")
                    continue
                key = (task, digest)
                if key in unique_seen or digest == baseline_sha[task]:
                    reject("duplicate_or_baseline")
                    continue
                unique_seen.add(key)
                try:
                    model = onnx.load(path, load_external_data=False)
                    onnx.checker.check_model(model, full_check=True)
                    ok, reasons, details = structural_audit(model, baselines[task])
                    if not ok:
                        for reason in reasons:
                            reject(reason)
                        continue
                    memory = static_memory(model)
                    parameter_count = params(model)
                    estimated_cost = memory + parameter_count
                except Exception as exc:
                    reject(f"parse_or_infer:{type(exc).__name__}")
                    continue
                if estimated_cost >= costs[task]:
                    reject("not_cheaper_static")
                    continue
                gain = math.log(costs[task] / estimated_cost) if estimated_cost else 25.0
                records.append({
                    "task": task,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest,
                    "serialized_size": path.stat().st_size,
                    "baseline_cost": costs[task],
                    "static_memory": memory,
                    "params": parameter_count,
                    "static_cost": estimated_cost,
                    "projected_gain": gain,
                    **details,
                })

    records.sort(key=lambda r: (-float(r["projected_gain"]), int(r["task"])))
    payload = {
        "baseline": "submission_base_8003.40.zip",
        "excluded_tasks": sorted(EXCLUDED_TASKS),
        "paths_seen": paths_seen,
        "size_prefiltered": size_prefiltered,
        "unique_models_seen": len(unique_seen),
        "retained": len(records),
        "reject_counts": dict(sorted(rejects.items())),
        "candidates": records,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({**{k: payload[k] for k in payload if k != "candidates"}, "top": records[:30]}, indent=2))


if __name__ == "__main__":
    main()
