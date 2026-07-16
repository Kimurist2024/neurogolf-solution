#!/usr/bin/env python3
"""Executable-vs-metadata diff audit for every A12 pending candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCAN = HERE / "retained_scan.json"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append("")
    return result


def attr_value(attr: onnx.AttributeProto) -> object:
    value = helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": sha_bytes(value.tobytes())}
    if isinstance(value, (tuple, list)):
        return list(value)
    if isinstance(value, onnx.TensorProto):
        arr = numpy_helper.to_array(value)
        return {"dtype": str(arr.dtype), "shape": list(arr.shape), "sha256": sha_bytes(arr.tobytes())}
    return value


def node_signature(node: onnx.NodeProto) -> dict[str, object]:
    return {
        "op_type": node.op_type,
        "domain": node.domain,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attributes": {attr.name: attr_value(attr) for attr in node.attribute},
    }


def initializer_signature(item: onnx.TensorProto) -> dict[str, object]:
    arr = numpy_helper.to_array(item)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "elements": int(arr.size),
        "sha256": sha_bytes(arr.tobytes()),
    }


def stripped_bytes(model: onnx.ModelProto) -> bytes:
    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    del clone.graph.value_info[:]
    clone.graph.name = ""
    clone.graph.doc_string = ""
    clone.doc_string = ""
    clone.producer_name = ""
    clone.producer_version = ""
    for node in clone.graph.node:
        node.name = ""
        node.doc_string = ""
    return clone.SerializeToString(deterministic=True)


def diff(base_path: Path, cand_path: Path) -> dict[str, object]:
    base = onnx.load(base_path)
    cand = onnx.load(cand_path)
    base_nodes = [node_signature(node) for node in base.graph.node]
    cand_nodes = [node_signature(node) for node in cand.graph.node]
    base_init = {item.name: initializer_signature(item) for item in base.graph.initializer}
    cand_init = {item.name: initializer_signature(item) for item in cand.graph.initializer}
    node_diffs = [
        {"index": index, "baseline": left, "candidate": right}
        for index, (left, right) in enumerate(zip(base_nodes, cand_nodes))
        if left != right
    ]
    if len(base_nodes) != len(cand_nodes):
        node_diffs.append({"node_count": [len(base_nodes), len(cand_nodes)]})
    init_diffs = [
        {"name": name, "baseline": base_init.get(name), "candidate": cand_init.get(name)}
        for name in sorted(set(base_init) | set(cand_init))
        if base_init.get(name) != cand_init.get(name)
    ]
    base_vi = {value.name: {"shape": dims(value), "elem_type": value.type.tensor_type.elem_type} for value in base.graph.value_info}
    cand_vi = {value.name: {"shape": dims(value), "elem_type": value.type.tensor_type.elem_type} for value in cand.graph.value_info}
    vi_diffs = [
        {"name": name, "baseline": base_vi.get(name), "candidate": cand_vi.get(name)}
        for name in sorted(set(base_vi) | set(cand_vi))
        if base_vi.get(name) != cand_vi.get(name)
    ]
    stripped_equal = stripped_bytes(base) == stripped_bytes(cand)
    return {
        "baseline": str(base_path.relative_to(ROOT)),
        "candidate": str(cand_path.relative_to(ROOT)),
        "baseline_sha256": sha_bytes(base_path.read_bytes()),
        "candidate_sha256": sha_bytes(cand_path.read_bytes()),
        "node_differences": node_diffs,
        "initializer_differences": init_diffs,
        "value_info_differences": vi_diffs,
        "graph_input_equal": base.graph.input == cand.graph.input,
        "graph_output_equal": base.graph.output == cand.graph.output,
        "opset_equal": base.opset_import == cand.opset_import,
        "clearing_metadata_makes_models_equal": stripped_equal,
        "metadata_only": stripped_equal and not node_diffs and not init_diffs,
    }


def main() -> None:
    scan = json.loads(SCAN.read_text())
    audits = []
    by_task: dict[int, list[Path]] = {}
    for row in scan["pending"]:
        task = int(row["task"])
        path = ROOT / row["path"]
        by_task.setdefault(task, []).append(path)
        audits.append(diff(HERE / "baseline" / f"task{task:03d}.onnx", path))
    peer = []
    for task, paths in sorted(by_task.items()):
        for index, left in enumerate(paths):
            for right in paths[index + 1:]:
                result = diff(left, right)
                result["task"] = task
                peer.append(result)
    (HERE / "diff_audit.json").write_text(json.dumps({"baseline_diffs": audits, "peer_diffs": peer}, indent=2) + "\n")


if __name__ == "__main__":
    main()
