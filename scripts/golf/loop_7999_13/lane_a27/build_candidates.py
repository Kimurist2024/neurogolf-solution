#!/usr/bin/env python3
"""Build only algebraically exact A27 probes; never promotes shared artifacts."""

from __future__ import annotations

from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent


def remove_task354_identity() -> Path:
    """Feed CenterCropPad from the unchanged initializer, removing Identity."""
    model = onnx.load(HERE / "base" / "task354.onnx")
    replacement = {"shape12_dyn": "target12"}
    kept = []
    for node in model.graph.node:
        if node.op_type == "Identity" and list(node.output) == ["shape12_dyn"]:
            continue
        for index, name in enumerate(node.input):
            if name in replacement:
                node.input[index] = replacement[name]
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    kept_vi = [value for value in model.graph.value_info if value.name != "shape12_dyn"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    path = HERE / "candidates" / "task354_no_identity.onnx"
    onnx.save(model, path)
    return path


def main() -> None:
    print(remove_task354_identity())


if __name__ == "__main__":
    main()
