#!/usr/bin/env python3
"""Build a compact period-fill ONNX for task017 (ARC 0dfd9992).

Rule (see reference_solver.py): restore black rectangular cutouts back to
the underlying periodic background pattern.  The pattern repeats every
`length` in [4,9].  We:

  1. decode the [1,10,30,30] one-hot input to a 30x30 float label grid
     (channel c contributes weight c), so black/out-of-grid -> 0.
  2. for every candidate period L in {4,...,9}:
       - compute a scalar "valid_L" flag = the L-shift agrees on all
         visible (non-zero) cells (out-of-grid 0s are excluded by the
         non-zero clean mask), and
       - compute fill_L = two sweeps of 4-direction copy-by-L
         (each black cell copies a non-zero donor L cells away).
  3. select fill for the smallest valid L via nested Where (priority
     4>5>6>7>8>9, fallback fill_9).
  4. mask to the 21x21 active region, then one-hot encode to
     [1,10,30,30] (out-of-grid + any residual 0 -> all channels 0).

Params are tiny (index/const tensors + a 10-vector decode weight); the
6 fill pipelines add compute, not parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "scripts/golf/scratch_wave/task017/cand.onnx"

H = W = 30
GRID = 21  # active region is rows/cols 0..20
L_CANDS = [4, 5, 6, 7, 8, 9]
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
ITERS = 2


class B:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.n = 0
        self._cache: dict[str, str] = {}

    def name(self, p: str) -> str:
        self.n += 1
        return f"{p}_{self.n}"

    def init(self, name: str, arr: np.ndarray) -> str:
        if name in self._cache:
            return name
        self.inits.append(numpy_helper.from_array(arr, name))
        self._cache[name] = name
        return name

    def node(self, op: str, ins: list[str], p: str, **attrs: object) -> str:
        out = self.name(p)
        self.nodes.append(helper.make_node(op, ins, [out], name=out, **attrs))
        return out


def shift_by(b: B, x: str, dr: int, dc: int, L: int) -> str:
    """Return x shifted so that out[i,j] = x[i+dr*L, j+dc*L], zero-padded.

    Implemented as Pad (on the source side) + Slice on the [1,1,30,30]
    label tensor.  dr,dc in {-1,0,1}.
    """
    sr, sc = dr * L, dc * L
    # We want result[i,j] = x[i+sr, j+sc].  Pad x then slice a 30x30 window.
    # pad_before/after on H and W so that the window [pb_h .. pb_h+30) maps to
    # source indices [-sr .. 30-sr).  Equivalent: pad both sides by L, then
    # slice starting at (L+sr, L+sc).
    pads = np.array(
        [0, 0, L, L, 0, 0, L, L], dtype=np.int64
    )  # [b,c,h_before,w_before, b,c,h_after,w_after] -> onnx order:
    # onnx Pad pads = [x1_begin,...,xn_begin, x1_end,...,xn_end]
    pads = np.array([0, 0, L, L, 0, 0, L, L], dtype=np.int64)
    padname = b.init(f"pad_L{L}", pads)
    padded = b.node("Pad", [x, padname, b.init("zero_f32", np.array(0.0, np.float32))], f"pad_{L}_{dr}_{dc}", mode="constant")
    h0, w0 = L + sr, L + sc
    starts = b.init(f"st_{L}_{dr}_{dc}", np.array([h0, w0], dtype=np.int64))
    ends = b.init(f"en_{L}_{dr}_{dc}", np.array([h0 + H, w0 + W], dtype=np.int64))
    axes = b.init("axes_hw", np.array([2, 3], dtype=np.int64))
    return b.node("Slice", [padded, starts, ends, axes], f"shift_{L}_{dr}_{dc}")


def fill_for_L(b: B, lab: str, L: int) -> str:
    zero = b.init("zero_f32", np.array(0.0, np.float32))
    out = lab
    for _ in range(ITERS):
        for dr, dc in DIRS:
            src = shift_by(b, out, dr, dc, L)
            is_black = b.node("Equal", [out, zero], f"blk_{L}")
            src_ok = b.node("Greater", [src, zero], f"srcok_{L}")
            take = b.node("And", [is_black, src_ok], f"take_{L}")
            out = b.node("Where", [take, src, out], f"fill_{L}")
    return out


def valid_for_L(b: B, lab: str, L: int) -> str:
    """Return a scalar bool: L-shift is consistent on all non-zero cells."""
    zero = b.init("zero_f32", np.array(0.0, np.float32))
    zero_i = b.init("zero_i64", np.array(0, np.int64))
    axes = b.init("axes_hw", np.array([2, 3], dtype=np.int64))

    diffs = []
    for dr, dc in [(0, 1), (1, 0)]:
        sr, sc = dr * L, dc * L
        # a = lab[:, :, :H-sr, :W-sc]; b_ = lab[:, :, sr:, sc:] (overlap region)
        st_a = b.init(f"va_st_{L}_{dr}", np.array([0, 0], np.int64))
        en_a = b.init(f"va_en_{L}_{dr}_{dc}", np.array([H - sr, W - sc], np.int64))
        a = b.node("Slice", [lab, st_a, en_a, axes], f"va_a_{L}")
        st_b = b.init(f"vb_st_{L}_{dr}_{dc}", np.array([sr, sc], np.int64))
        en_b = b.init(f"vb_en_{L}", np.array([H, W], np.int64))
        bb = b.node("Slice", [lab, st_b, en_b, axes], f"va_b_{L}")
        ne = b.node("Equal", [a, bb], f"va_eq_{L}")  # equal-> we invert
        neq = b.node("Not", [ne], f"va_ne_{L}")
        a_nz = b.node("Greater", [a, zero], f"va_anz_{L}")
        b_nz = b.node("Greater", [bb, zero], f"va_bnz_{L}")
        both = b.node("And", [a_nz, b_nz], f"va_both_{L}")
        bad = b.node("And", [neq, both], f"va_bad_{L}")
        bad_i = b.node("Cast", [bad], f"va_badi_{L}", to=TensorProto.INT64)
        cnt = b.node("ReduceSum", [bad_i], f"va_cnt_{L}", keepdims=0)
        diffs.append(cnt)
    total = b.node("Add", diffs, f"va_tot_{L}")
    valid = b.node("Equal", [total, zero_i], f"va_valid_{L}")  # scalar bool
    return valid


def main() -> None:
    b = B()

    # ---- decode one-hot -> float label grid [1,1,30,30] ----
    w_decode = b.init("w_decode", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    lab = b.node("Conv", ["input", w_decode], "lab")  # [1,1,30,30]

    # ---- per-L fills and validity flags ----
    fills = {L: fill_for_L(b, lab, L) for L in L_CANDS}
    valids = {L: valid_for_L(b, lab, L) for L in L_CANDS}

    # ---- select smallest valid L (priority 4>5>...>9, fallback fill_9) ----
    res = fills[9]
    for L in [8, 7, 6, 5, 4]:
        res = b.node("Where", [valids[L], fills[L], res], f"sel_{L}")

    # ---- mask to 21x21 active region ----
    # Out-of-grid cells (rows/cols >= 21) must encode to ALL channels 0.
    # We set them to a sentinel label (-1) that matches no channel 0..9.
    # In-grid cells keep their filled value (always 1..9 after fill).
    region = np.zeros((1, 1, H, W), dtype=np.float32)
    region[0, 0, :GRID, :GRID] = 1.0
    inside = b.init("inside", region)
    inside_bool = b.node("Greater", [inside, b.init("zero_f32", np.array(0.0, np.float32))], "inside_bool")
    sentinel = b.init("sentinel_f32", np.array(-1.0, np.float32))
    masked = b.node("Where", [inside_bool, res, sentinel], "masked")

    # ---- one-hot encode: output[c] = (label == c) ----
    arange = b.init("arange_f32", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    out = b.node("Equal", [masked, arange], "onehot")  # [1,10,30,30] bool
    b.nodes[-1].output[0] = "output"

    graph = helper.make_graph(
        b.nodes,
        "task017_fill",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        b.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=10
    )
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
