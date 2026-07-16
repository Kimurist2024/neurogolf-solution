#!/usr/bin/env python3
"""Inventory the latest cost-26..50 authority models without mutating the submission."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
SCORES = ROOT / "all_scores.csv"
OUTPUT = HERE / "inventory.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def attr_value(attribute: onnx.AttributeProto) -> Any:
    value = helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, onnx.TensorProto):
        return {
            "dtype": onnx.TensorProto.DataType.Name(value.data_type),
            "shape": list(value.dims),
            "values": numpy_helper.to_array(value).reshape(-1)[:64].tolist(),
        }
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def summarize_initializer(item: onnx.TensorProto) -> dict[str, Any]:
    array = numpy_helper.to_array(item)
    flat = array.reshape(-1)
    finite = np.isfinite(flat) if np.issubdtype(array.dtype, np.number) else np.ones(flat.shape, bool)
    unique: list[Any] = []
    if flat.size <= 256:
        try:
            unique = np.unique(flat).tolist()
        except Exception:  # noqa: BLE001
            unique = []
    return {
        "name": item.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "elements": int(array.size),
        "finite": bool(finite.all()),
        "minimum": float(np.min(flat[finite])) if np.any(finite) else None,
        "maximum": float(np.max(flat[finite])) if np.any(finite) else None,
        "unique": unique[:64],
        "values": flat[:128].tolist(),
    }


def main() -> int:
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA256 mismatch")
    costs: dict[int, int] = {}
    for line in SCORES.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split(",")
        if len(fields) >= 4:
            costs[int(fields[1][4:])] = int(fields[3])
    tasks = sorted(task for task, cost in costs.items() if 26 <= cost <= 50)
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in tasks:
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_from_string(data)
            rows.append({
                "task": task,
                "cost": costs[task],
                "sha256": sha256(data),
                "file_bytes": len(data),
                "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
                "nodes": [
                    {
                        "index": index,
                        "op": node.op_type,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                        "attributes": {attribute.name: attr_value(attribute) for attribute in node.attribute},
                    }
                    for index, node in enumerate(model.graph.node)
                ],
                "initializers": [summarize_initializer(item) for item in model.graph.initializer],
            })
    OUTPUT.write_text(json.dumps({"authority": str(AUTHORITY.relative_to(ROOT)), "tasks": rows}, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(rows)} tasks)")
    for row in rows:
        init = "; ".join(
            f"{item['name']}:{item['dtype']}{item['shape']}={item['elements']}"
            for item in row["initializers"]
        )
        ops = " -> ".join(node["op"] for node in row["nodes"])
        print(f"task{row['task']:03d} cost={row['cost']:2d} ops={ops:<32} init={init}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
