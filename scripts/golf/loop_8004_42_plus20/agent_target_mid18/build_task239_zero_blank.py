#!/usr/bin/env python3
"""Build the true-rule task239 decoder with a literal zero blank feature.

The incumbent computes ``rows - max_count`` for every off-bar cell.  Its final
QLinearConv has input zero-point 0, so the literal all-zero feature already
maps exactly to an all-zero output vector.  Generator colors are 1..9 and the
inactive TopK slots are separately suppressed by ``safe_truefeat``.
"""

from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
BASE = (
    HERE.parent / "agent_exact_wave2" / "base" / "task239.onnx"
)
OUT = HERE / "task239_zero_blank.onnx"


def main() -> None:
    model = onnx.load(BASE)
    kept = []
    for node in model.graph.node:
        if node.output and node.output[0] in {"maxcount", "dist"}:
            continue
        if node.output and node.output[0] == "feat":
            assert node.op_type == "Where" and node.input[2] == "dist"
            node.input[2] = "zero8"
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
