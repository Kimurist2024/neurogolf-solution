#!/usr/bin/env python3
"""Probe exact CenterCropPad-chain bypasses in the sound task118 baseline."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "lane_task118_crop_fusion"
BASE = ROOT / "submission_base_7999.13.zip"
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402


VARIANTS: dict[str, dict[str, tuple[str, str]]] = {
    "x8_direct": {"x8c_25x27": ("x8c_26x27", "x8")},
    "u_direct": {"u": ("u_tmp", "u0")},
    "cond_direct": {"cond": ("condp_29x30", "cond_core")},
    "u0_direct": {"cond_core": ("u0_hw", "u0")},
    "cw_direct": {"centers": ("cw_dyn", "cw")},
    "x8_u": {
        "x8c_25x27": ("x8c_26x27", "x8"),
        "u": ("u_tmp", "u0"),
    },
    "u0_cond": {
        "cond_core": ("u0_hw", "u0"),
        "cond": ("condp_29x30", "cond_core"),
    },
    "all_data_chains": {
        "x8c_25x27": ("x8c_26x27", "x8"),
        "u": ("u_tmp", "u0"),
        "cond_core": ("u0_hw", "u0"),
        "cond": ("condp_29x30", "cond_core"),
        "centers": ("cw_dyn", "cw"),
    },
}


def prune(model: onnx.ModelProto) -> None:
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    required_values = {item.name for item in model.graph.output}
    required_nodes: set[int] = set()
    stack = list(required_values)
    while stack:
        value = stack.pop()
        node = producers.get(value)
        if node is None or id(node) in required_nodes:
            continue
        required_nodes.add(id(node))
        stack.extend(name for name in node.input if name)
    kept_nodes = [node for node in model.graph.node if id(node) in required_nodes]
    used = {name for node in kept_nodes for name in node.input if name}
    kept_initializers = [item for item in model.graph.initializer if item.name in used]
    produced = {name for node in kept_nodes for name in node.output if name}
    kept_vi = [item for item in model.graph.value_info if item.name in produced or item.name in used]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)


def build(source: onnx.ModelProto, edits: dict[str, tuple[str, str]]) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(source)
    seen: set[str] = set()
    for node in model.graph.node:
        if not node.output or node.output[0] not in edits:
            continue
        old, new = edits[node.output[0]]
        replacements = 0
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
                replacements += 1
        if replacements != 1:
            raise ValueError(f"{node.output[0]} expected one {old}, got {replacements}")
        seen.add(node.output[0])
    if seen != set(edits):
        raise ValueError(f"missing outputs: {set(edits) - seen}")
    prune(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def score(model: onnx.ModelProto, label: str) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory(prefix="task118_fuse_") as directory:
        result = scoring.score_and_verify(
            model, 118, directory, label=label, require_correct=True
        )
    if result is None:
        return None
    return {
        "memory": int(result["memory"]),
        "params": int(result["params"]),
        "cost": int(result["cost"]),
        "correct": bool(result["correct"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE) as archive:
        source = onnx.load_model_from_string(archive.read("task118.onnx"))
    rows: list[dict[str, object]] = []
    baseline = score(source, "baseline")
    for name, edits in VARIANTS.items():
        row: dict[str, object] = {"name": name, "edits": edits}
        try:
            candidate = build(source, edits)
            path = OUT / f"task118_{name}.onnx"
            onnx.save(candidate, path)
            row["path"] = str(path.relative_to(ROOT))
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["node_count"] = len(candidate.graph.node)
            row["initializer_count"] = len(candidate.graph.initializer)
            row["score"] = score(candidate, name)
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)
        rows.append(row)
        print(json.dumps(row))
    payload = {"baseline": baseline, "variants": rows}
    (OUT / "screen.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
