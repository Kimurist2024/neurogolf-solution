#!/usr/bin/env python3
"""Compact EXACT period-tile reconstruction ONNX for task017 (ARC 0dfd9992).

Rule: restore black rectangular cutouts to the underlying periodic
background pattern  value(row,col) = ((g(row)^2+g(col)^2) % mod) + 1,
g(i) = (offset+i) % length - length//2,  length in [4,9].

Single-pipeline, period-complete reconstruction (0 fails / 120000 fresh):
  1. crop one-hot input to 21x21, decode -> uint8 label grid (black/oob = 0).
  2. for each candidate L in {4..9}: a scalar valid_L flag (L-shift agrees on
     every visible cell) and a folded residue tile placed top-left in a 9x9
     (ReduceMax over the period classes; black=0, so per-class max = pattern
     value, or 0 if the whole class is cut out).
  3. pick the SMALLEST valid L: select its 9x9 tile AND its precomputed
     residue->tile index map (both via nested Where over the 6 scalar flags;
     fallback L=9).
  4. close the tile under the pattern symmetries: transpose symmetry + ONE
     row-equivalence merge (rows agreeing where co-defined and sharing a
     defined column donate values).  This recovers fully-cut-out classes.
  5. back-project with a single Gather(tile_flat, index_map) -> 21x21, then
     substitute where the input was black.
  6. pad to 30x30 with sentinel 255 (out-of-grid -> all channels 0), one-hot.

All compute is uint8 on <=24x24 / 9x9 tensors, so the runtime memory
footprint (sum of all intermediate tensor bytes) is small.
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
TILE = 9  # pad every residue tile into a 9x9 (top-left LxL)
L_CANDS = [4, 5, 6, 7, 8, 9]
U8 = TensorProto.UINT8


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
    Z = b.init("zero_u8", np.array(0, np.uint8))
    ax01 = b.init("ax01", np.array([0, 1], np.int64))
    st0 = b.init("st0", np.array([0, 0], np.int64))
    full = b.init("full", np.array([GRID, GRID], np.int64))

    # ---- decode one-hot -> 1-channel label, then crop to 21x21 active region ----
    # Conv first (output 1 channel) avoids slicing the 10-channel float input.
    axhw = b.init("axhw", np.array([2, 3], np.int64))
    w_decode = b.init("w_decode", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    lab_f = b.node("Conv", ["input", w_decode], "lab_f")  # [1,1,30,30]
    lab_u8 = b.node("Cast", [lab_f], "lab_u8", to=U8)
    lab_g = b.node("Slice", [lab_u8, st0, b.init("enG_4", np.array([GRID, GRID], np.int64)), axhw], "lab_g")  # [1,1,21,21]
    g2d = b.node("Reshape", [lab_g, b.init("rsp2d", np.array([GRID, GRID], np.int64))], "g2d")  # [21,21]

    Z255 = b.init("u255", np.array(255, np.uint8))

    def tile_and_valid(L: int) -> tuple[str, str]:
        # Fold the 21x21 grid into L x L residue classes, then derive BOTH the
        # per-class max tile and a scalar validity flag from the same fold.
        Hp = ((GRID + L - 1) // L) * L
        nL = Hp // L
        pa = Hp - GRID
        gp = b.node("Pad", [g2d, b.init(f"pads_{L}", np.array([0, 0, pa, pa], np.int64)), Z], f"gp_{L}", mode="constant")
        g4 = b.node("Reshape", [gp, b.init(f"rsp4_{L}", np.array([nL, L, nL, L], np.int64))], f"g4_{L}")
        # per-class max (pattern value, or 0 if class fully cut/padding)
        t1 = b.node("ReduceMax", [g4], f"tm0_{L}", axes=[0], keepdims=0)
        cmax = b.node("ReduceMax", [t1], f"tm1_{L}", axes=[1], keepdims=0)  # [L,L]
        # validity: a class is consistent iff its non-zero members are all equal,
        # i.e. min-over-nonzero == max.  Replace 0 with 255 before ReduceMin.
        g4nz = b.node("Where", [b.node("Equal", [g4, Z], f"g4z_{L}"), Z255, g4], f"g4nz_{L}")
        m1 = b.node("ReduceMin", [g4nz], f"cm0_{L}", axes=[0], keepdims=0)
        cmin = b.node("ReduceMin", [m1], f"cm1_{L}", axes=[1], keepdims=0)  # [L,L]
        has = b.node("Greater", [cmax, Z], f"has_{L}")               # class has a value
        diff = b.node("Not", [b.node("Equal", [cmin, cmax], f"eqmm_{L}")], f"diff_{L}")
        conflict = b.node("And", [has, diff], f"conf_{L}")           # [L,L]
        anyconf = b.node("ReduceMax", [b.node("Cast", [conflict], f"confu_{L}", to=U8)], f"anyc_{L}", keepdims=0)
        valid = b.node("Equal", [anyconf, Z], f"vvalid_{L}")         # scalar bool
        tile9 = b.node("Pad", [cmax, b.init(f"pad9_{L}", np.array([0, 0, TILE - L, TILE - L], np.int64)), Z], f"tile9_{L}", mode="constant")
        return tile9, valid

    tv = {L: tile_and_valid(L) for L in L_CANDS}
    tiles = {L: tv[L][0] for L in L_CANDS}
    valids = {L: tv[L][1] for L in L_CANDS}

    # select smallest valid L's tile (priority 4>...>9, fallback 9)
    tile = tiles[9]
    Lsel = b.init("L9_i32", np.array(9, np.int32))  # scalar fallback
    for L in [8, 7, 6, 5, 4]:
        tile = b.node("Where", [valids[L], tiles[L], tile], f"seltile_{L}")
        Lsel = b.node("Where", [valids[L], b.init(f"L{L}_i32", np.array(L, np.int32)), Lsel], f"selL_{L}")

    # build the residue->tile index map from the scalar Lsel:
    #   imap[r][c] = (r % Lsel) * 9 + (c % Lsel)
    rows_i = b.init("rows_i", np.arange(GRID, dtype=np.int32).reshape(GRID, 1))
    cols_i = b.init("cols_i", np.arange(GRID, dtype=np.int32).reshape(1, GRID))
    rmod = b.node("Mod", [rows_i, Lsel], "rmod", fmod=0)  # [21,1]
    cmod = b.node("Mod", [cols_i, Lsel], "cmod", fmod=0)  # [1,21]
    rscaled = b.node("Mul", [rmod, b.init("tile_w", np.array(TILE, np.int32))], "rscaled")
    imap = b.node("Add", [rscaled, cmod], "imap")  # [21,21] int32, broadcast

    # ---- close tile under symmetry: transpose + one row-equivalence merge ----
    tile = b.node("Max", [tile, b.node("Transpose", [tile], "tT", perm=[1, 0])], "tsym")
    A = b.node("Reshape", [tile, b.init("rA", np.array([TILE, 1, TILE], np.int64))], "A")
    Bt = b.node("Reshape", [tile, b.init("rB", np.array([1, TILE, TILE], np.int64))], "Bt")
    ne = b.node("Not", [b.node("Equal", [A, Bt], "meq")], "mne")
    both = b.node("And", [b.node("Greater", [A, Z], "manz"), b.node("Greater", [Bt, Z], "mbnz")], "mboth")
    disagree = b.node("And", [ne, both], "mdis")
    rowbad = b.node("ReduceMax", [b.node("Cast", [disagree], "mdc", to=U8)], "mrowbad", axes=[2], keepdims=0)
    shareany = b.node("ReduceMax", [b.node("Cast", [both], "mbc", to=U8)], "mshare", axes=[2], keepdims=0)
    nrb = b.node("Not", [b.node("Greater", [rowbad, Z], "mrbb")], "mnrb")
    compat = b.node("And", [nrb, b.node("Greater", [shareany, Z], "mshb")], "mcompat")
    compat = b.node("Or", [compat, b.init("eye9", np.eye(TILE, dtype=bool))], "mcompatd")
    compat_u = b.node("Cast", [compat], "mcu", to=U8)
    comp3 = b.node("Reshape", [compat_u, b.init("rc", np.array([TILE, TILE, 1], np.int64))], "comp3")
    tile3 = b.node("Reshape", [tile, b.init("rt", np.array([1, TILE, TILE], np.int64))], "tile3")
    donor = b.node("Mul", [comp3, tile3], "donor")  # [9,9,9]
    tile = b.node("ReduceMax", [donor], "newt", axes=[1], keepdims=0)  # [9,9]

    # ---- back-project via Gather(tile_flat, imap) ----
    tflat = b.node("Reshape", [tile, b.init("rflat", np.array([TILE * TILE], np.int64))], "tflat")
    reps = b.node("Gather", [tflat, imap], "reps", axis=0)  # [21,21]
    isblk = b.node("Equal", [g2d, Z], "isblk")
    res = b.node("Where", [isblk, reps, g2d], "res")  # [21,21]

    # pad to 30x30 with sentinel 255, one-hot
    res30 = b.node("Pad", [res, b.init("padsB", np.array([0, 0, H - GRID, W - GRID], np.int64)), b.init("sent255", np.array(255, np.uint8))], "res30", mode="constant")
    res4 = b.node("Reshape", [res30, b.init("rsp4d", np.array([1, 1, H, W], np.int64))], "res4")
    arange = b.init("arange_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
    b.node("Equal", [res4, arange], "onehot")
    b.nodes[-1].output[0] = "output"

    graph = helper.make_graph(
        b.nodes,
        "task017_tile",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        b.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
