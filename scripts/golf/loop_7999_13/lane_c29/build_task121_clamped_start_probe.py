#!/usr/bin/env python3
"""Build a cost-124 task121 probe; it is deliberately not promotion-safe."""

from __future__ import annotations

from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task121.onnx"
OUTPUT = HERE / "task121_clamped_start_probe.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    # Slice axis 1 has extent 10.  Its incumbent start -12 clamps to 0, so the
    # existing int8 zero scalar is an exact replacement at the Slice boundary.
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "neg12_i8":
                node.input[index] = "i8_zero"
    kept = [item for item in model.graph.initializer if item.name != "neg12_i8"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
