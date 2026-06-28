#!/usr/bin/env python3
"""Param-cut rebuild: take the cost-9216 incumbent and replace its 4 expensive
Conv detector kernels (1200 params) with param-light reduce equivalents.

The 4 Convs computed:
  line_row_kernel    : sum cyan(ch8) along WIDTH  -> [1,1,30,1]  (full-row test)
  line_col_kernel    : sum cyan(ch8) along HEIGHT -> [1,1,1,30]  (full-col test)
  seed_row_plus1_*   : sum red(ch2)*(r+1) along HEIGHT -> [1,1,1,30] per col (seed row+1)
  seed_col_plus1_*   : sum red(ch2)*(c+1) along WIDTH  -> [1,1,30,1] per row (seed col+1)

Replace with: slice the single channel, multiply by a small arange weight
(30 params for seed maps; 0 for line counts), ReduceSum. Output dtypes/shapes
kept bit-identical to the original Conv outputs so the rest of the graph is
untouched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / "artifacts" / "handcrafted" / "task379.onnx"
OUT = HERE / "cand.onnx"

CYAN = 8
RED = 2


def main() -> None:
    m = onnx.load(str(SRC))
    g = m.graph

    # Names of the Conv outputs we will reproduce (the *_30f float tensors).
    conv_outputs = {
        "row_line_30f": ("cyan_count_w", CYAN, "w", None),       # sum cyan over width
        "col_line_30f": ("cyan_count_h", CYAN, "h", None),       # sum cyan over height
        "seed_row_p1_h_30f": ("seed_rowp1", RED, "h", "rowp1"),  # sum red*(r+1) over height
        "seed_col_p1_v_30f": ("seed_colp1", RED, "w", "colp1"),  # sum red*(c+1) over width
    }

    # Replace ONLY the two LINE detector Convs with param-free reduces (saves
    # 600 params, zero memory penalty: reduce-first keeps intermediates tiny).
    # Keep the two SEED-position Convs unchanged — replacing them would force a
    # [1,1,30,30] intermediate that costs more memory than the params it saves.
    LINE_CONV_OUTS = {"row_line_30f", "col_line_30f"}
    conv_kernels = {"line_row_kernel", "line_col_kernel"}
    new_nodes = [n for n in g.node
                 if not (n.op_type == "Conv" and any(o in LINE_CONV_OUTS for o in n.output))]
    new_inits = [it for it in g.initializer if it.name not in conv_kernels]

    # Helper to append a node.
    extra_nodes = []
    extra_inits = []

    def add_init(name, arr):
        extra_inits.append(numpy_helper.from_array(arr, name))
        return name

    # constants
    add_init("_zero2", np.array([0], np.int64))
    add_init("_one2", np.array([1], np.int64))
    # arange(1..30) as float32, for height [1,1,30,1] and width [1,1,1,30]
    ar_h = (np.arange(1, 31, dtype=np.float32)).reshape(1, 1, 30, 1)
    ar_w = (np.arange(1, 31, dtype=np.float32)).reshape(1, 1, 1, 30)
    add_init("_arange_h", ar_h)
    add_init("_arange_w", ar_w)

    chan_a = add_init("_ca1", np.array([1], np.int64))

    def ch_slice_init(ci: int, tag: str):
        s = add_init(f"_cs_{tag}", np.array([ci], np.int64))
        e = add_init(f"_ce_{tag}", np.array([ci + 1], np.int64))
        return s, e

    # ---- LINE detectors: reduce the full input FIRST (cheap [1,10,*,*]=1200B),
    #      then slice channel 8. No weighting, no [1,1,30,30] intermediate. ----
    ax_w = add_init("_axw", np.array([3], np.int64))   # reduce width -> [1,10,30,1]
    ax_h = add_init("_axh", np.array([2], np.int64))   # reduce height -> [1,10,1,30]
    cs8, ce8 = ch_slice_init(CYAN, "cy")
    # cyan over width -> row_line_30f
    extra_nodes.append(helper.make_node("ReduceSum", ["input", ax_w], ["_cyw10"], name="rs_cyw", keepdims=1))
    extra_nodes.append(helper.make_node("Slice", ["_cyw10", cs8, ce8, chan_a], ["row_line_30f"], name="sl_cyw"))
    # cyan over height -> col_line_30f
    extra_nodes.append(helper.make_node("ReduceSum", ["input", ax_h], ["_cyh10"], name="rs_cyh", keepdims=1))
    extra_nodes.append(helper.make_node("Slice", ["_cyh10", cs8, ce8, chan_a], ["col_line_30f"], name="sl_cyh"))

    # SEED maps keep their original Conv kernels (left untouched in new_nodes).

    # Prepend our producing nodes (they only depend on 'input' + inits), keep graph valid order.
    final_nodes = extra_nodes + new_nodes

    new_graph = helper.make_graph(
        final_nodes, g.name, list(g.input), list(g.output),
        new_inits + extra_inits,
    )
    m2 = helper.make_model(new_graph, opset_imports=list(m.opset_import))
    m2.ir_version = m.ir_version
    onnx.checker.check_model(m2, full_check=True)
    onnx.save(m2, str(OUT))
    n_params = sum(int(np.prod(it.dims)) if it.dims else 1 for it in m2.graph.initializer)
    print(f"saved {OUT}  params={n_params}")


if __name__ == "__main__":
    main()
