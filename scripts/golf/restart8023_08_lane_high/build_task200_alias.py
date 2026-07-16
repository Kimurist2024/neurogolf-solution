#!/usr/bin/env python3
"""Alias task200's duplicate [0,1] OneHot values to the Conv bias."""

from __future__ import annotations

from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "focus/candidates/task200_POLICY90_cost342_c659ae401e4c.onnx"
OUTPUT = HERE / "task200_alias_oh_values_to_conv_bias.onnx"


def main() -> int:
    model = onnx.load(SOURCE)
    onehot = next(node for node in model.graph.node if node.op_type == "OneHot")
    if list(onehot.input) != ["seed_f", "depth", "oh_values"]:
        raise RuntimeError(f"unexpected OneHot inputs: {list(onehot.input)}")
    onehot.input[2] = "conv_bias"
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    if any(item.name == "oh_values" for item in model.graph.initializer):
        raise RuntimeError("duplicate initializer was not removed")
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
