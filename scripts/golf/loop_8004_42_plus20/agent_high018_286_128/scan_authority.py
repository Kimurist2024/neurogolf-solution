#!/usr/bin/env python3
"""Inventory current 8009.46 authority members for task018/task286.

This lane is intentionally non-promoting.  It emits only evidence and candidate
members below this directory; root submission and ledgers are never written.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TASKS = (18, 286)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCAN = load_module(
    "high128_scan_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/scan_candidates.py",
)
AUDIT = load_module(
    "high128_audit_tools",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_8009_exact_B_116/audit_candidates.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_trace(task: int, data: bytes) -> dict[str, object]:
    try:
        return AUDIT.direct_trace(task, data)
    except Exception as exc:  # noqa: BLE001 - a failed strict trace is evidence
        return {
            "truthful": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def json_attr(attribute: onnx.AttributeProto):
    value = helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, onnx.TensorProto):
        return np.asarray(numpy_helper.to_array(value)).tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def graph_detail(model: onnx.ModelProto) -> dict[str, object]:
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(index)
    return {
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
        "nodes": [
            {
                "index": index,
                "name": node.name,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {attribute.name: json_attr(attribute) for attribute in node.attribute},
                "output_consumers": {
                    name: consumers.get(name, []) for name in node.output if name
                },
            }
            for index, node in enumerate(model.graph.node)
        ],
        "initializers": [
            {
                "name": item.name,
                "dtype": TensorProto.DataType.Name(item.data_type),
                "shape": list(item.dims),
                "elements": int(np.asarray(numpy_helper.to_array(item)).size),
                "values": np.asarray(numpy_helper.to_array(item)).tolist(),
                "consumers": consumers.get(item.name, []),
            }
            for item in model.graph.initializer
        ],
    }


def main() -> int:
    authority_data = AUTHORITY.read_bytes()
    if digest(authority_data) != AUTHORITY_SHA256:
        raise RuntimeError("submission.zip changed from immutable 8009.46 authority")
    current = HERE / "current"
    current.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks": {},
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = {task: archive.read(f"task{task:03d}.onnx") for task in TASKS}
    for task, data in members.items():
        path = current / f"task{task:03d}.onnx"
        path.write_bytes(data)
        model = onnx.load_model_from_string(data)
        report["tasks"][str(task)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "serialized_bytes": len(data),
            "official_profile": SCAN.official_cost(data, f"high128_task{task:03d}_current"),
            "structural": SCAN.structural(copy.deepcopy(model)),
            "runtime_shape_trace": safe_trace(task, data),
            "inventory": SCAN.graph_inventory(copy.deepcopy(model)),
            "graph_detail": graph_detail(model),
        }
        (HERE / f"task{task:03d}_graph.txt").write_text(
            onnx.printer.to_text(model) + "\n", encoding="utf-8"
        )
        row = report["tasks"][str(task)]
        print(
            f"task{task:03d} sha={row['sha256'][:12]} "
            f"cost={row['official_profile']['cost']} "
            f"structural={row['structural'].get('pass')} "
            f"truthful={row['runtime_shape_trace'].get('truthful')}",
            flush=True,
        )
    (HERE / "authority_inventory.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
