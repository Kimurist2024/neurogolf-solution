#!/usr/bin/env python3
"""Minimal ONNX for task379 (ecdecbb3) — cyan-line drip rendering.

Validated rule (band-fill closed form, no iteration):
  Input has full-row OR full-column cyan lines + red seeds. Each seed drips red
  toward nearest line each direction; a between-two-lines run fills entirely,
  outer runs fill seed->line; a 3x3 cyan box stamps where a drip meets a line,
  center stays red.

Efficiency: everything is BOOL/UINT8 (1 byte) except the 3 matmul triples that
must be float. Single uint8 LABEL plane; Equal->output (bool) is free. Pipeline
runs ONCE on a canonical orientation (branchless bool select on whether full
cyan ROWS exist), result transposed back. 20x20 window; out-of-grid sentinel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
OUT = HERE / "cand.onnx"

N = 20
CYAN = 8
RED = 2
PERM = [0, 1, 3, 2]


class B:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.k = 0

    def nm(self, p: str) -> str:
        self.k += 1
        return f"{p}_{self.k}"

    def init(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def nd(self, op: str, ins: list[str], p: str, **attrs) -> str:
        o = self.nm(p)
        self.nodes.append(helper.make_node(op, ins, [o], name=o, **attrs))
        return o


def pipeline(b: B, cyan_b: str, red_b: str, ingrid_b: str) -> tuple[str, str]:
    """cyan_b/red_b/ingrid_b: [1,1,N,N] bool. Returns (cyan_out, red_out) bool."""
    zf = b.zf
    cyan_f = b.nd("Cast", [cyan_b], "cyan_f", to=TensorProto.FLOAT)
    ingrid_f = b.nd("Cast", [ingrid_b], "ingrid_f", to=TensorProto.FLOAT)

    # line rows: full cyan rows within grid
    crow = b.nd("ReduceSum", [cyan_f, b.axW], "crow", keepdims=1)
    irow = b.nd("ReduceSum", [ingrid_f, b.axW], "irow", keepdims=1)
    line_col_b = b.nd("And", [b.nd("Equal", [crow, irow], "eq"), b.nd("Greater", [crow, zf], "cpos")], "linecol")  # [1,1,N,1] bool
    lineV = b.nd("Reshape", [b.nd("Cast", [line_col_b], "linef", to=TensorProto.FLOAT), b.shN], "lineV")  # [N] f32

    cumline = b.nd("MatMul", [b.Ltri, lineV], "cumline")          # [N]
    notlineV_b = b.nd("Not", [b.nd("Reshape", [line_col_b, b.shN], "lineVb")], "notlineV_b")  # [N] bool

    cl_col = b.nd("Reshape", [cumline, b.shN1], "clcol")
    cl_row = b.nd("Reshape", [cumline, b.sh1N], "clrow")
    same = b.nd("Equal", [cl_col, cl_row], "same")               # [N,N] bool
    nl_col = b.nd("Reshape", [notlineV_b, b.shN1], "nlcol")
    nl_row = b.nd("Reshape", [notlineV_b, b.sh1N], "nlrow")
    nl_mat = b.nd("And", [nl_col, nl_row], "nlmat")              # [N,N]

    Rdown = b.nd("Cast", [b.nd("And", [b.nd("And", [same, b.tri_le], "sd"), nl_mat], "Rdownb")], "Rdown", to=TensorProto.FLOAT)
    Rup = b.nd("Cast", [b.nd("And", [b.nd("And", [same, b.tri_ge], "su"), nl_mat], "Rupb")], "Rup", to=TensorProto.FLOAT)
    redM = b.nd("Reshape", [red_b, b.shNN], "redM_b")
    redM_f = b.nd("Cast", [redM], "redM_f", to=TensorProto.FLOAT)
    down = b.nd("Greater", [b.nd("MatMul", [Rdown, redM_f], "downf"), zf], "down")
    up = b.nd("Greater", [b.nd("MatMul", [Rup, redM_f], "upf"), zf], "up")

    cumline_excl = b.nd("Sub", [cumline, lineV], "clex")
    aa = b.nd("Reshape", [b.nd("Greater", [cumline_excl, zf], "aav"), b.shN1], "aa")  # [N,1] bool
    tot = b.nd("ReduceSum", [lineV, b.axN0], "tot", keepdims=1)
    ab = b.nd("Reshape", [b.nd("Greater", [b.nd("Sub", [tot, cumline], "below"), zf], "abv"), b.shN1], "ab")
    band0 = b.nd("Or", [b.nd("And", [down, ab], "db"), b.nd("And", [up, aa], "ua")], "band0")
    nl_col2 = b.nd("Reshape", [notlineV_b, b.shN1], "nlcol2")
    redband = b.nd("And", [band0, nl_col2], "redband")          # [N,N] bool
    rb4 = b.nd("Reshape", [redband, b.sh4], "rb4")             # [1,1,N,N] bool

    # box centers: line[r] & (redband[r-1] | redband[r+1])  (bool shift via Pad/Slice on uint8)
    rb_u8 = b.nd("Cast", [rb4], "rb_u8", to=TensorProto.UINT8)
    up_sh = b.nd("Slice", [b.nd("Pad", [rb_u8, b.padup], "uppad"), b.s0, b.sN, b.ax2], "upsh")
    dn_sh = b.nd("Slice", [b.nd("Pad", [rb_u8, b.paddn], "dnpad"), b.s1, b.sN1, b.ax2], "dnsh")
    adj = b.nd("Greater", [b.nd("Add", [up_sh, dn_sh], "adj"), b.z_u8], "adjb")  # [1,1,N,N] bool
    line4b = b.nd("Reshape", [line_col_b, b.sh4col], "line4b")  # [1,1,N,1] bool
    bc = b.nd("And", [adj, line4b], "bc")                      # [1,1,N,N] bool

    # 3x3 dilate via MaxPool on uint8
    bc_u8 = b.nd("Cast", [bc], "bc_u8", to=TensorProto.UINT8)
    box = b.nd("Greater", [b.nd("MaxPool", [bc_u8], "boxf", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]), b.z_u8], "box")

    not_box = b.nd("Not", [box], "notbox")
    not_bc = b.nd("Not", [bc], "notbc")
    cyan_out = b.nd("And", [b.nd("Or", [cyan_b, box], "cob"), not_bc], "cyan_out")
    red_out = b.nd("Or", [b.nd("And", [rb4, not_box], "rbnb"), bc], "red_out")
    return cyan_out, red_out


def bool_select(b: B, a: str, c: str, cond_b: str, ncond_b: str, nm: str) -> str:
    """branchless: cond_b ? a : c  with bool tensors (cond broadcast [1,1,1,1])."""
    return b.nd("Or", [b.nd("And", [a, cond_b], f"{nm}_a"), b.nd("And", [c, ncond_b], f"{nm}_b")], f"{nm}_sel")


def main() -> None:
    b = B()
    b.zf = b.init("zf", np.array(0.0, dtype=np.float32))
    b.z_u8 = b.init("z_u8", np.array(0, dtype=np.uint8))
    b.axW = b.init("axW", np.array([3], np.int64))
    b.axN0 = b.init("axN0", np.array([0], np.int64))
    b.ax2 = b.init("ax2", np.array([2], np.int64))
    b.shN = b.init("shN", np.array([N], np.int64))
    b.shN1 = b.init("shN1", np.array([N, 1], np.int64))
    b.sh1N = b.init("sh1N", np.array([1, N], np.int64))
    b.shNN = b.init("shNN", np.array([N, N], np.int64))
    b.sh4 = b.init("sh4", np.array([1, 1, N, N], np.int64))
    b.sh4col = b.init("sh4col", np.array([1, 1, N, 1], np.int64))
    b.Ltri = b.init("Ltri", np.tril(np.ones((N, N), np.float32)))
    idx = np.arange(N)
    b.tri_le = b.init("trile", (idx[:, None] >= idx[None, :]))
    b.tri_ge = b.init("trige", (idx[:, None] <= idx[None, :]))
    b.padup = b.init("padup", np.array([0, 0, 1, 0, 0, 0, 0, 0], np.int64))
    b.paddn = b.init("paddn", np.array([0, 0, 0, 0, 0, 0, 1, 0], np.int64))
    b.s0 = b.init("s0", np.array([0], np.int64))
    b.sN = b.init("sN", np.array([N], np.int64))
    b.s1 = b.init("s1", np.array([1], np.int64))
    b.sN1 = b.init("sN1", np.array([N + 1], np.int64))

    hw_s = b.init("hw_s", np.array([0, 0], np.int64))
    hw_e = b.init("hw_e", np.array([N, N], np.int64))
    hw_ax = b.init("hw_ax", np.array([2, 3], np.int64))

    def chan_bool(ci: int, name: str) -> str:
        c1 = b.nd("Slice", ["input", b.init(f"cs_{name}", np.array([ci], np.int64)),
                            b.init(f"ce_{name}", np.array([ci + 1], np.int64)),
                            b.init(f"ca_{name}", np.array([1], np.int64))], f"chc_{name}")
        cw = b.nd("Slice", [c1, hw_s, hw_e, hw_ax], f"ch_{name}")
        return b.nd("Greater", [cw, b.zf], f"chb_{name}")  # bool [1,1,N,N]

    cyan0 = chan_bool(CYAN, "cyan")
    red0 = chan_bool(RED, "red")
    ingrid30 = b.nd("ReduceMax", ["input", b.init("axC", np.array([1], np.int64))], "ingrid30", keepdims=1)
    ingrid0 = b.nd("Greater", [b.nd("Slice", [ingrid30, hw_s, hw_e, hw_ax], "ingrid0f"), b.zf], "ingrid0")  # bool

    # orientation: any full cyan ROW?
    cyan0_f = b.nd("Cast", [cyan0], "cyan0_f", to=TensorProto.FLOAT)
    ingrid0_f = b.nd("Cast", [ingrid0], "ingrid0_f", to=TensorProto.FLOAT)
    crow = b.nd("ReduceSum", [cyan0_f, b.axW], "o_crow", keepdims=1)
    irow = b.nd("ReduceSum", [ingrid0_f, b.axW], "o_irow", keepdims=1)
    o_line = b.nd("And", [b.nd("Equal", [crow, irow], "o_eq"), b.nd("Greater", [crow, b.zf], "o_cpos")], "o_line")  # [1,1,N,1]
    o_line_f = b.nd("Cast", [o_line], "o_line_f", to=TensorProto.FLOAT)
    rowsum = b.nd("ReduceSum", [o_line_f, b.ax2], "o_rowsum", keepdims=1)  # [1,1,1,1]
    has_h = b.nd("Greater", [rowsum, b.zf], "has_h")   # [1,1,1,1] bool
    not_h = b.nd("Not", [has_h], "not_h")

    # canonicalize (bool select): if not has_h, transpose
    def canon(t: str, nm: str) -> str:
        tt = b.nd("Transpose", [t], f"{nm}_T", perm=PERM)
        return bool_select(b, t, tt, has_h, not_h, f"{nm}_c")

    cyan_c = canon(cyan0, "cyan")
    red_c = canon(red0, "red")
    ingrid_c = canon(ingrid0, "ingrid")

    cy_m, rd_m = pipeline(b, cyan_c, red_c, ingrid_c)

    # de-canonicalize
    def uncanon(t: str, nm: str) -> str:
        tt = b.nd("Transpose", [t], f"{nm}_uT", perm=PERM)
        return bool_select(b, t, tt, has_h, not_h, f"{nm}_u")

    cyan_out = uncanon(cy_m, "cyo")    # [1,1,N,N] bool
    red_out = uncanon(rd_m, "rdo")

    # ---- single uint8 LABEL plane ----
    # color = cyan?8 : red?2 : ingrid?0 : (handled by pad sentinel)
    # in-grid non-colored = 0 already; out-of-grid via pad sentinel 10.
    cyan_u8 = b.nd("Cast", [cyan_out], "cyan_u8", to=TensorProto.UINT8)
    red_u8 = b.nd("Cast", [red_out], "red_u8", to=TensorProto.UINT8)
    # red contributes 2 only where not cyan; cyan contributes 8.
    red_only = b.nd("Mul", [red_u8, b.nd("Cast", [b.nd("Not", [cyan_out], "not_cy")], "not_cy_u8", to=TensorProto.UINT8)], "red_only")
    c2 = b.nd("Mul", [red_only, b.init("two_u8", np.array(2, np.uint8))], "c2")
    c8 = b.nd("Mul", [cyan_u8, b.init("eight_u8", np.array(8, np.uint8))], "c8")
    color_colored = b.nd("Add", [c2, c8], "color_colored")   # {0,2,8} within window

    # out-of-grid cells inside the window must be sentinel 10 (not black 0).
    # color = ingrid ? color_colored : 10  == color_colored + (1-ingrid)*10
    notgrid_u8 = b.nd("Cast", [b.nd("Not", [ingrid0], "notgrid")], "notgrid_u8", to=TensorProto.UINT8)
    oog10 = b.nd("Mul", [notgrid_u8, b.init("ten10_u8", np.array(10, np.uint8))], "oog10")
    color_u8 = b.nd("Add", [color_colored, oog10], "color_u8")   # {0,2,8} in-grid, 10 out

    color30 = b.nd("Pad", [color_u8, b.init("padHW", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64)),
                           b.init("ten_u8", np.array(10, np.uint8))], "color30")
    channel_ids = b.init("channel_ids", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
    b.nodes.append(helper.make_node("Equal", [color30, channel_ids], ["output"], name="final_out"))

    graph = helper.make_graph(
        b.nodes, "task379",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        b.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(OUT))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
