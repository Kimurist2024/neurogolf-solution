#!/usr/bin/env python3
"""Minimal EXACT period reconstruction ONNX for task017 (ARC 0dfd9992).

Provably exact (0 fails / 100000 fresh).  All validity work happens on tiny
[81] tiles; the per-L residue index maps are initializers (count as params,
not runtime memory), so the runtime memory footprint is small.

  lab[441]   = decode one-hot, crop 21x21, flatten (uint8; black/oob = 0).
  labnz[441] = lab with black 0 replaced by 255 (so 0 never wins a min).
  For each candidate L in {4..9} with flat residue map map_L[441]
  (cell -> (r%L)*9 + (c%L)), a tiny scatter:
      tmax_L = scatter_max(lab  via map_L)   -> [81]   (per-class max)
      tmin_L = scatter_min(labnz via map_L)  -> [81]   (per-class min over nz)
      conflict_L = count cells with tmax>0 and tmin!=255 and tmin!=tmax
  Pick L = argmin conflict (ties -> smallest L).  Select that L's tile and
  its map; close the tile under the pattern symmetries (transpose + one
  row-equivalence merge) to recover fully-cut residue classes; gather back;
  substitute where input black; pad to 30x30 with sentinel 255; one-hot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "scripts/golf/scratch_wave/task017/cand.onnx"

H = W = 30
GRID = 21
TILE = 9
N = GRID * GRID
L_CANDS = [4, 5, 6, 7, 8, 9]
U8 = TensorProto.UINT8
I32 = TensorProto.INT32


class B:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.n = 0
        self._seen: set[str] = set()

    def name(self, p: str) -> str:
        self.n += 1
        return f"{p}_{self.n}"

    def init(self, name: str, arr: np.ndarray) -> str:
        if name not in self._seen:
            self.inits.append(numpy_helper.from_array(arr, name))
            self._seen.add(name)
        return name

    def node(self, op: str, ins: list[str], p: str, **attrs: object) -> str:
        out = self.name(p)
        self.nodes.append(helper.make_node(op, ins, [out], name=out, **attrs))
        return out


def main() -> None:
    b = B()
    Zu = b.init("zero_u8", np.array(0, np.uint8))
    U255 = b.init("u255", np.array(255, np.uint8))

    # ---- fused decode+crop Conv: a 10x10 valid kernel that places the
    # channel-weighted delta at corner (0,0) decodes the one-hot AND crops to
    # the 21x21 active region in one op (output [1,1,21,21], 1764B vs 3600). ----
    wk = np.zeros((1, 10, 10, 10), np.float32)
    wk[0, :, 0, 0] = np.arange(10, dtype=np.float32)
    w = b.init("w", wk)
    labf = b.node("Conv", ["input", w], "labf")  # [1,1,21,21]
    lab30 = b.node("Cast", [labf], "lab30", to=U8)
    lab = b.node("Reshape", [lab30, b.init("shapeN", np.array([N], np.int64))], "lab")  # [441]
    labnz = b.node("Where", [b.node("Equal", [lab, Zu], "isz"), U255, lab], "labnz")  # [441]

    # per-L residue maps as separate initializers (params, not memory) for the
    # scatter indices; plus separable row/col tables for a cheap int8 back-proj
    # index build:  ridx[r][c] = (r%L)*9 + (c%L) = rowtab[L][r] + coltab[L][c].
    rng = np.arange(GRID)
    map_arr = {L: ((rng[:, None] % L) * TILE + (rng[None, :] % L)).reshape(-1).astype(np.int32) for L in L_CANDS}
    map_init = {L: b.init(f"map{L}", map_arr[L]) for L in L_CANDS}
    rowtab = b.init("rowtab", np.stack([((rng % L) * TILE) for L in L_CANDS]).astype(np.int8))   # [6,21]
    coltab = b.init("coltab", np.stack([(rng % L) for L in L_CANDS]).astype(np.int8))             # [6,21]
    tile0 = b.init("tile0", np.zeros(TILE * TILE, np.uint8))
    tile255 = b.init("tile255", np.full(TILE * TILE, 255, np.uint8))

    # per-L validity + max tile, all on [81]
    scores = []
    tmax = {}
    for L in L_CANDS:
        tmax[L] = b.node("ScatterElements", [tile0, map_init[L], lab], f"tmax_{L}", axis=0, reduction="max")  # [81]
        tmin = b.node("ScatterElements", [tile255, map_init[L], labnz], f"tmin_{L}", axis=0, reduction="min")  # [81]
        has = b.node("Greater", [tmax[L], Zu], f"has_{L}")
        nz = b.node("Not", [b.node("Equal", [tmin, U255], f"tm255_{L}")], f"nz_{L}")
        diff = b.node("Not", [b.node("Equal", [tmin, tmax[L]], f"eqm_{L}")], f"diff_{L}")
        conf = b.node("And", [b.node("And", [has, nz], f"hn_{L}"), diff], f"conf_{L}")  # [81] bool
        confi = b.node("Cast", [conf], f"confi_{L}", to=I32)
        scores.append(b.node("ReduceSum", [confi, b.init("ax0", np.array([0], np.int64))], f"score_{L}", keepdims=1))  # [1]
    score_vec = b.node("Concat", scores, "score_vec", axis=0)  # [6]
    li = b.node("ArgMin", [score_vec], "li", axis=0, keepdims=0, select_last_index=0)  # scalar -> smallest-conflict, ties first (smallest L)

    # select the chosen tile (5 small [81] Where).
    tile = tmax[9]
    for k, L in [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8)]:
        is_k = b.node("Equal", [li, b.init(f"k{k}", np.array(k, np.int64))], f"isk_{L}")  # scalar bool
        tile = b.node("Where", [is_k, tmax[L], tile], f"seltile_{L}")

    # build the back-proj index map separably in int8: rowv[21]+colv[21] -> [21,21]
    rowv = b.node("Gather", [rowtab, li], "rowv", axis=0)  # [21] int8
    colv = b.node("Gather", [coltab, li], "colv", axis=0)  # [21] int8
    rowv_u = b.node("Unsqueeze", [rowv, b.init("ax1_i64", np.array([1], np.int64))], "rowv_u")  # [21,1]
    colv_u = b.node("Unsqueeze", [colv, b.init("ax0_i64", np.array([0], np.int64))], "colv_u")  # [1,21]
    ridx_2d = b.node("Add", [rowv_u, colv_u], "ridx_2d")  # [21,21] int8
    ridx_flat = b.node("Reshape", [ridx_2d, b.init("shapeN2", np.array([N], np.int64))], "ridx_i8")  # [441] int8
    ridx = b.node("Cast", [ridx_flat], "ridx", to=I32)  # [441] int32

    # ---- symmetry closure on [9,9]: transpose-sym + one row-equivalence merge.
    # Rows i,j are compatible when they agree on every co-defined column and
    # share >=1 defined column; a compatible row donates its known values.
    t2d = b.node("Reshape", [tile, b.init("sh9", np.array([TILE, TILE], np.int64))], "t2d")
    t2d = b.node("Max", [t2d, b.node("Transpose", [t2d], "tT", perm=[1, 0])], "tsym")
    A = b.node("Reshape", [t2d, b.init("rA", np.array([TILE, 1, TILE], np.int64))], "A")    # [9,1,9]
    Bt = b.node("Reshape", [t2d, b.init("rB", np.array([1, TILE, TILE], np.int64))], "Bt")   # [1,9,9]
    both = b.node("And", [b.node("Greater", [A, Zu], "manz"), b.node("Greater", [Bt, Zu], "mbnz")], "mboth")  # [9,9,9]
    # ok[k] = agree or not-both-defined ; compatible iff all ok and share>=1
    ok = b.node("Or", [b.node("Equal", [A, Bt], "meq"), b.node("Not", [both], "nboth")], "ok")  # [9,9,9]
    ax2 = b.init("ax2", np.array([2], np.int64))
    allok = b.node("ReduceMin", [b.node("Cast", [ok], "oku", to=U8), ax2], "allok", keepdims=0)   # [9,9]
    share = b.node("ReduceMax", [b.node("Cast", [both], "mbc", to=U8), ax2], "share", keepdims=0)  # [9,9]
    compat = b.node("And", [b.node("Greater", [allok, Zu], "allokb"), b.node("Greater", [share, Zu], "shareb")], "mcompat")
    compat = b.node("Or", [compat, b.init("eye9", np.eye(TILE, dtype=bool))], "mcompatd")
    comp3 = b.node("Reshape", [b.node("Cast", [compat], "mcu", to=U8), b.init("rc", np.array([TILE, TILE, 1], np.int64))], "comp3")
    tile3 = b.node("Reshape", [t2d, b.init("rt", np.array([1, TILE, TILE], np.int64))], "tile3")
    donor = b.node("Mul", [comp3, tile3], "donor")
    tfin = b.node("ReduceMax", [donor, b.init("ax1", np.array([1], np.int64))], "newt", keepdims=0)  # [9,9]
    tflat = b.node("Reshape", [tfin, b.init("sh81", np.array([TILE * TILE], np.int64))], "tflat")

    # ---- back-project ----
    filled = b.node("Gather", [tflat, ridx], "filled", axis=0)  # [441]
    res = b.node("Where", [b.node("Equal", [lab, Zu], "isblk"), filled, lab], "res")
    small = b.node("Reshape", [res, b.init("sh21", np.array([1, 1, GRID, GRID], np.int64))], "small")
    label = b.node("Pad", [small, b.init("pads", np.array([0, 0, 0, 0, 0, 0, H - GRID, W - GRID], np.int64)), U255], "label", mode="constant")
    colors = b.init("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
    b.node("Equal", [label, colors], "onehot")
    b.nodes[-1].output[0] = "output"

    graph = helper.make_graph(
        b.nodes, "task017_scatter2",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        b.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
