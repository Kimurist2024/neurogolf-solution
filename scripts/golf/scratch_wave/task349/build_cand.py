#!/usr/bin/env python3
"""Minimal ONNX for task349 (db93a21d, 'death stars').

Rule (compiled): input one-hot [1,10,30,30], only maroon(9). Maroon squares are
2r x 2r (r in 1..5), bottom row r, left col c.
  - maroon(9) out == maroon in.
  - green(3) halo = square dilated by r. With A_t = "maroon cell in a horizontal
    run of length >= 2t" (= radius >= t), green = UNION_t dilate_t(A_t). Computed
    cumulatively: acc=0; for t=5..1: acc = dilate1(acc) | A_t; green = dilate1(acc).
    (gives dilate_t(A_t) for each t with kernel-3 pools only.)
  - blue(1) beam = square columns, all rows strictly below = (T_lower @ M) > 0.
  - black(0) = in-grid & no color; in-grid = sum over input channels > 0.
  Priority maroon > green > blue.

Dtypes: morphology + masks in uint8 (MaxPool/Min/Max support uint8; AND=Min,
OR=Max). The only logical NOT (no uint8 Sub) is the per-width erosion complement
and a few compose complements, done in bool. OOB treated as non-maroon via
Pad(value=1) before the horizontal erosion MaxPool.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
OUT = HERE / "cand.onnx"
H = W = 30
R_MAX = 5
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
F32 = TensorProto.FLOAT


class B:
    def __init__(self) -> None:
        self.nodes: list = []
        self.inits: list = []
        self.n = 0

    def name(self, p: str) -> str:
        self.n += 1
        return f"{p}_{self.n}"

    def init(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def nd(self, op: str, ins: list, p: str, **attrs) -> str:
        out = self.name(p)
        self.nodes.append(helper.make_node(op, ins, [out], name=out, **attrs))
        return out

    def NOT_u8(self, x_u8: str, p: str) -> str:
        """uint8 0/1 logical NOT via bool round-trip."""
        xb = self.nd("Cast", [x_u8], p + "_b", to=BOOL)
        nb = self.nd("Not", [xb], p + "_nb")
        return self.nd("Cast", [nb], p + "_u", to=U8)


def main() -> None:
    b = B()
    c_s = b.init("c9s", np.array([9], dtype=np.int64))
    c_e = b.init("c9e", np.array([10], dtype=np.int64))
    c_a = b.init("c9a", np.array([1], dtype=np.int64))
    Msl = b.nd("Slice", ["input", c_s, c_e, c_a], "Msl")        # f32 [1,1,30,30]
    Mu = b.nd("Cast", [Msl], "Mu", to=U8)                       # uint8 maroon
    notMu = b.NOT_u8(Mu, "notM")
    one_u8 = b.init("one_u8", np.array(1, dtype=np.uint8))

    def Aopen(width: int) -> str:
        """uint8 mask: maroon cells in a horizontal run of length >= width."""
        pad_amt = b.init(f"pad{width}",
                         np.array([0, 0, 0, 0, 0, 0, 0, width - 1], dtype=np.int64))
        notM_p = b.nd("Pad", [notMu, pad_amt, one_u8], f"nMp{width}", mode="constant")
        erp = b.nd("MaxPool", [notM_p], f"erp{width}",
                   kernel_shape=[1, width], pads=[0, 0, 0, 0], strides=[1, 1])
        erode = b.NOT_u8(erp, f"erode{width}")
        cov = b.nd("MaxPool", [erode], f"cov{width}",
                   kernel_shape=[1, width], pads=[0, width - 1, 0, 0], strides=[1, 1])
        return cov

    A = {t: Aopen(2 * t) for t in range(1, R_MAX + 1)}

    # cumulative dilation: acc=0; t=5..1: acc = dilate1(acc) | A_t ; green=dilate1(acc)
    acc = None
    for t in range(R_MAX, 0, -1):
        if acc is None:
            acc = A[t]
        else:
            d = b.nd("MaxPool", [acc], f"dil{t}",
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
            acc = b.nd("Max", [d, A[t]], f"acc{t}")
    green_u = b.nd("MaxPool", [acc], "green_u",
                   kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])

    # in-grid mask (uint8) via sum over channels > 0.
    sum_axis = b.init("sumax", np.array([1], dtype=np.int64))
    grid_f = b.nd("ReduceSum", ["input", sum_axis], "grid_f", keepdims=1)
    grid_u = b.nd("Cast", [grid_f], "grid_u", to=U8)            # f32>0 -> 1

    # blue beam via strictly-lower-triangular MatMul over rows (f32).
    T = np.tril(np.ones((H, H), dtype=np.float32), k=-1)
    T_init = b.init("T_lower", T)
    beam_raw = b.nd("MatMul", [T_init, Msl], "beam_raw")        # f32
    zero_f = b.init("zero_f", np.array(0.0, dtype=np.float32))
    beam_b = b.nd("Greater", [beam_raw, zero_f], "beam_b")
    beam_u = b.nd("Cast", [beam_b], "beam_u", to=U8)

    # compose (uint8): AND=Min, OR=Max, NOT via bool.
    notMu2 = notMu
    not_green = b.NOT_u8(green_u, "ngrn")
    green_only = b.nd("Min", [green_u, notMu2], "green_only")   # green & ~maroon
    green_g = b.nd("Min", [green_only, grid_u], "green_g")      # & grid
    blue1 = b.nd("Min", [beam_u, notMu2], "blue1")             # beam & ~maroon
    blue2 = b.nd("Min", [blue1, not_green], "blue2")          # & ~green
    blue_g = b.nd("Min", [blue2, grid_u], "blue_g")           # & grid

    # black(0) = grid & ~(maroon | green_g | blue_g)
    anyc = b.nd("Max", [green_g, blue_g], "anyc")
    anyc2 = b.nd("Max", [anyc, Mu], "anyc2")
    no_color = b.NOT_u8(anyc2, "nocol")
    ch0 = b.nd("Min", [grid_u, no_color], "ch0")

    zeros_ch = b.init("zeros_ch", np.zeros((1, 1, H, W), dtype=np.uint8))
    parts = [ch0, blue_g, zeros_ch, green_g, zeros_ch, zeros_ch,
             zeros_ch, zeros_ch, zeros_ch, Mu]
    b.nodes.append(helper.make_node("Concat", parts, ["output"], name="out_concat",
                                    axis=1))

    inp = helper.make_tensor_value_info("input", F32, [1, 10, H, W])
    outp = helper.make_tensor_value_info("output", U8, [1, 10, H, W])
    graph = helper.make_graph(b.nodes, "task349", [inp], [outp], b.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(OUT))
    print("saved", OUT)


if __name__ == "__main__":
    main()
