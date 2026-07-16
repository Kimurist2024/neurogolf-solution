#!/usr/bin/env python3
"""Build task250 with the spec-derived ROI kept in fp32 end-to-end.

The immutable 8004.50 graph computes exact generator box coordinates as fp32,
casts them through integer/fp16 carriers, and casts the assembled ROI back to
fp32 for Resize.  This rebuild keeps the already-computed fp32 coordinates for
the ROI branch while retaining the integer branch used by row/column masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task250.onnx"
def build(direct_axes: set[str], output: Path) -> None:
    model = onnx.load(SOURCE)
    removed = {"roi4_f"}
    if "r" in direct_axes:
        removed.add("r_beg_f16")
    if "c" in direct_axes:
        removed.add("c_beg_f16")
    kept = []
    for node in model.graph.node:
        if any(name in removed for name in node.output):
            continue
        if node.output and node.output[0] == "r_start_f16" and "r" in direct_axes:
            node.input[0] = "r_pos"
            node.output[0] = "r_start_f"
        elif node.output and node.output[0] == "c_start_f16" and "c" in direct_axes:
            node.input[0] = "c_pos"
            node.output[0] = "c_start_f"
        elif node.output and node.output[0] == "r_end_f16" and "r" in direct_axes:
            node.input[0] = "r_start_f"
            node.output[0] = "r_end_f"
        elif node.output and node.output[0] == "c_end_f16" and "c" in direct_axes:
            node.input[0] = "c_start_f"
            node.output[0] = "c_end_f"
        elif node.output and node.output[0] == "roi4_f16":
            node.input[:] = [
                "r_start_f" if "r" in direct_axes else "r_start_f16",
                "c_start_f" if "c" in direct_axes else "c_start_f16",
                "r_end_f" if "r" in direct_axes else "r_end_f16",
                "c_end_f" if "c" in direct_axes else "c_end_f16",
            ]
            node.output[0] = "roi4_f"
        elif node.op_type == "Resize":
            node.input[1] = "roi4_f"
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axes", choices=("r", "c", "rc"), default="rc")
    args = parser.parse_args()
    output = HERE / f"task250_direct_roi_{args.axes}.onnx"
    build(set(args.axes), output)


if __name__ == "__main__":
    main()
