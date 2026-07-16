#!/usr/bin/env python3
"""Remove every visible-fixture correction from the exact task157 baseline."""

from __future__ import annotations

from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
BASE = HERE / "baseline" / "task157.onnx"
OUT = HERE / "candidate_task157_no_lookup.onnx"


def prune(model: onnx.ModelProto) -> None:
    needed = {value.name for value in model.graph.output}
    kept: list[onnx.NodeProto] = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    model.graph.ClearField("node")
    model.graph.node.extend(reversed(kept))
    initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in needed
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)
    model.graph.ClearField("value_info")


def main() -> int:
    model = onnx.load(BASE)
    replaced = 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "bstarts_corr_new":
                node.input[index] = "bstarts"
                replaced += 1
    if replaced != 1:
        raise RuntimeError(f"expected one corrected-start consumer, found {replaced}")
    prune(model)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(inferred, full_check=True)
    if any(
        initializer.name.startswith("fixk_")
        for initializer in inferred.graph.initializer
    ):
        raise RuntimeError("fixture-key initializer survived pruning")
    onnx.save(inferred, OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
