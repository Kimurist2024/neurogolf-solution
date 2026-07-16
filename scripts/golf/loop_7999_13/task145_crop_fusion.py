#!/usr/bin/env python3
"""Probe mathematically centered crop/pad fusion in the sound task145 model."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "lane_task145_crop_fusion"
SOURCE = HERE / "lane_baseline_fresh100/models/task145.onnx"
sys.path.insert(0, str(ROOT / "scripts"))

from lib import scoring  # noqa: E402


onnxruntime.set_default_logger_severity(3)

VARIANTS: dict[str, dict[str, tuple[str, str]]] = {
    "hw_direct": {"wc_20": ("hw_21", "xb")},
    "pad_direct": {"code30": ("code29", "code20")},
    "cc5_direct": {"cc_5": ("cc_6", "wc_20")},
    "cc2_direct": {"cc_2": ("cc_3", "wc_20")},
    "wall_direct": {"wall_b": ("cc_5", "wc_20")},
    "bg_direct": {"bg_b": ("cc_2", "wc_20")},
    "channels_direct": {
        "wall_b": ("cc_5", "wc_20"),
        "bg_b": ("cc_2", "wc_20"),
    },
    "spatial_direct": {
        "wc_20": ("hw_21", "xb"),
        "code30": ("code29", "code20"),
    },
    "all_direct": {
        "wc_20": ("hw_21", "xb"),
        "wall_b": ("cc_5", "wc_20"),
        "bg_b": ("cc_2", "wc_20"),
        "code30": ("code29", "code20"),
    },
}


def prune(model: onnx.ModelProto) -> None:
    producers = {output: node for node in model.graph.node for output in node.output if output}
    required_nodes: set[int] = set()
    stack = [item.name for item in model.graph.output]
    while stack:
        node = producers.get(stack.pop())
        if node is None or id(node) in required_nodes:
            continue
        required_nodes.add(id(node))
        stack.extend(name for name in node.input if name)
    kept_nodes = [node for node in model.graph.node if id(node) in required_nodes]
    used = {name for node in kept_nodes for name in node.input if name}
    kept_initializers = [item for item in model.graph.initializer if item.name in used]
    names = used | {name for node in kept_nodes for name in node.output if name}
    kept_vi = [item for item in model.graph.value_info if item.name in names]
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
        for output in node.output:
            if output not in edits:
                continue
            old, new = edits[output]
            count = 0
            for index, value in enumerate(node.input):
                if value == old:
                    node.input[index] = new
                    count += 1
            if count != 1:
                raise ValueError(f"{output}: expected one {old}, got {count}")
            seen.add(output)
    if seen != set(edits):
        raise ValueError(f"missing {set(edits) - seen}")
    prune(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def score(model: onnx.ModelProto, label: str) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory(prefix="task145_fuse_") as directory:
        result = scoring.score_and_verify(model, 145, directory, label=label, require_correct=True)
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
    source = onnx.load(SOURCE)
    baseline = score(source, "baseline")
    rows: list[dict[str, object]] = []
    for name, edits in VARIANTS.items():
        row: dict[str, object] = {"name": name, "edits": edits}
        try:
            candidate = build(source, edits)
            path = OUT / f"task145_{name}.onnx"
            onnx.save(candidate, path)
            row.update(
                path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                node_count=len(candidate.graph.node),
                initializer_count=len(candidate.graph.initializer),
                score=score(candidate, name),
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)
        rows.append(row)
        print(json.dumps(row))
    (OUT / "screen.json").write_text(
        json.dumps({"baseline": baseline, "variants": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
