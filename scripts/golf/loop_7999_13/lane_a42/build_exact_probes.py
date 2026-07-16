#!/usr/bin/env python3
"""Build the remaining local type-template eliminations for task196."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline_task196.onnx"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_unused_initializer(model: onnx.ModelProto, name: str) -> None:
    if any(name in node.input for node in model.graph.node):
        return
    kept = [init for init in model.graph.initializer if init.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def build(label: str, replacement: onnx.NodeProto) -> dict[str, object]:
    model = copy.deepcopy(onnx.load(SOURCE))
    target_index = next(
        index
        for index, node in enumerate(model.graph.node)
        if node.output == ["g_bool"]
    )
    model.graph.node[target_index].CopyFrom(replacement)
    remove_unused_initializer(model, "bool_zero")
    checker = strict = True
    error = None
    try:
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:
        checker = strict = False
        error = repr(exc)
    path = HERE / f"probe_{label}.onnx"
    onnx.save(model, path)
    return {
        "label": label,
        "path": path.name,
        "sha256": sha(path),
        "full_checker": checker,
        "strict_shape_data_prop": strict,
        "error": error,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }


def main() -> None:
    probes = [
        build(
            "g_bool_cast",
            helper.make_node("Cast", ["g_raw"], ["g_bool"], name="ToBool", to=onnx.TensorProto.BOOL),
        ),
        build(
            "g_bool_greater",
            helper.make_node("Greater", ["g_raw", "zero_u8"], ["g_bool"], name="ToBool"),
        ),
    ]
    (HERE / "exact_probe_manifest.json").write_text(
        json.dumps({"source_sha256": sha(SOURCE), "probes": probes}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
