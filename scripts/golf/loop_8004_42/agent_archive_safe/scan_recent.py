#!/usr/bin/env python3
"""Read-only structural screen for ONNX artifacts newer than the all400 scan."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "scripts/golf/loop_8004_42/submission_8004.42_fixed_rebase_meta.zip"
CUTOFF = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json"
OUT = HERE / "recent_structural_leads.json"
LOOSE_RE = re.compile(r"^task(\d{3})(?:[^0-9].*)?\.onnx$", re.IGNORECASE)
MEMBER_RE = re.compile(r"(?:^|/)task(\d{3})\.onnx$", re.IGNORECASE)
FIXED = {
    13, 15, 20, 31, 68, 71, 79, 88, 105, 109, 132, 158, 174, 183,
    189, 206, 221, 240, 243, 259, 300, 302, 344, 349, 358, 379, 398,
}
EXCLUDED = FIXED | {
    9, 73, 76, 77, 96, 101, 112, 118, 134, 153, 168, 173, 185, 192,
    196, 198, 201, 208, 219, 251, 273, 286, 322, 323, 333, 343, 372,
    382, 391, 396,
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def costs() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open(newline="") as handle:
        return {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }


def static_cost(model: onnx.ModelProto) -> int | None:
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        initializers = {item.name for item in inferred.graph.initializer}
        params = sum(int(np.prod(item.dims, dtype=np.int64)) for item in inferred.graph.initializer)
        values = {
            item.name: item
            for item in list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        }
        memory = 0
        for name in {name for node in inferred.graph.node for name in node.output if name}:
            if name in {"input", "output"} or name in initializers:
                continue
            value = values.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                return None
            tensor = value.type.tensor_type
            if not tensor.HasField("shape"):
                return None
            elements = 1
            for dim in tensor.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0:
                    return None
                elements *= int(dim.dim_value)
            memory += elements * np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type)).itemsize
        return int(params + memory)
    except Exception:
        return None


def gate(data: bytes) -> tuple[int | None, str]:
    if len(data) > 1_440_000:
        return None, "oversized"
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:
        return None, f"checker_or_shape:{type(exc).__name__}"
    if model.functions or model.graph.sparse_initializer:
        return None, "function_or_sparse"
    if any(item.domain not in {"", "ai.onnx"} for item in model.opset_import):
        return None, "nonstandard_domain"
    for initializer in model.graph.initializer:
        if initializer.external_data or initializer.data_location == onnx.TensorProto.EXTERNAL:
            return None, "external_initializer"
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED or "SEQUENCE" in upper:
            return None, f"banned:{node.op_type}"
        if node.op_type == "TfIdfVectorizer":
            return None, "lookup_tfidf"
        if node.op_type == "Einsum" and len(node.input) >= 15:
            return None, f"giant_einsum:{len(node.input)}"
        if any(attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS} for attr in node.attribute):
            return None, "nested_graph"
    # Exact initializer-backed bias lengths. Dynamic biases are not eligible.
    init = {item.name: item for item in model.graph.initializer}
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {
        value.name: [int(dim.dim_value) if dim.HasField("dim_value") else 0 for dim in value.type.tensor_type.shape.dim]
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
        if value.type.HasField("tensor_type")
    }
    for node in model.graph.node:
        bias_index = 8 if node.op_type == "QLinearConv" else (2 if node.op_type in {"Conv", "ConvTranspose"} else None)
        if bias_index is None or len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        bias = init.get(node.input[bias_index])
        output_shape = shapes.get(node.output[0], [])
        if bias is None or len(output_shape) < 2 or output_shape[1] <= 0:
            return None, "dynamic_or_unknown_conv_bias"
        if int(np.prod(bias.dims, dtype=np.int64)) != output_shape[1]:
            return None, "short_conv_bias"
    value = static_cost(model)
    return value, "pass" if value is not None else "unknown_cost"


def main() -> None:
    cutoff = CUTOFF.stat().st_mtime
    current_cost = costs()
    with zipfile.ZipFile(BASE) as archive:
        base_hash = {task: sha(archive.read(f"task{task:03d}.onnx")) for task in range(1, 401)}
    unique: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)

    def add(task: int, data: bytes, source: str) -> None:
        if task in EXCLUDED:
            return
        digest = sha(data)
        if digest == base_hash[task]:
            return
        row = unique[task].setdefault(digest, {"data": data, "sources": []})
        row["sources"].append(source)

    roots = (ROOT / "others", ROOT / "scripts/golf")
    files_seen = 0
    for source_root in roots:
        for path in source_root.rglob("*"):
            if not path.is_file() or path.stat().st_mtime <= cutoff or HERE in path.parents:
                continue
            if path.suffix.lower() == ".onnx":
                match = LOOSE_RE.match(path.name)
                if match:
                    files_seen += 1
                    try:
                        add(int(match.group(1)), path.read_bytes(), str(path.relative_to(ROOT)))
                    except OSError:
                        pass
            elif path.suffix.lower() == ".zip":
                files_seen += 1
                try:
                    with zipfile.ZipFile(path) as archive:
                        for name in archive.namelist():
                            match = MEMBER_RE.search(name)
                            if match:
                                add(int(match.group(1)), archive.read(name), f"{path.relative_to(ROOT)}::{name}")
                except Exception:
                    pass

    leads = []
    rejects: dict[str, int] = defaultdict(int)
    for task, candidates in sorted(unique.items()):
        for digest, row in candidates.items():
            value, reason = gate(row["data"])
            if value is None or value >= current_cost[task]:
                rejects[reason if value is None else "not_cheaper"] += 1
                continue
            leads.append({
                "task": task,
                "current_cost": current_cost[task],
                "static_cost": value,
                "sha256": digest,
                "sources": row["sources"][:20],
            })
    leads.sort(key=lambda row: (row["static_cost"] - row["current_cost"], row["task"]))
    OUT.write_text(json.dumps({
        "baseline": str(BASE.relative_to(ROOT)),
        "cutoff": str(CUTOFF.relative_to(ROOT)),
        "files_seen": files_seen,
        "unique_candidates": sum(map(len, unique.values())),
        "rejections": dict(rejects),
        "leads": leads,
    }, indent=2) + "\n")
    print(f"files={files_seen} unique={sum(map(len, unique.values()))} leads={len(leads)}")
    for row in leads:
        print(f"task{row['task']:03d} {row['current_cost']}->{row['static_cost']} {row['sources'][0]}")


if __name__ == "__main__":
    main()
