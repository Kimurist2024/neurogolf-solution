#!/usr/bin/env python3
"""Build single-staircase + 1D-outside ONNX for task382 (archetype apply_gravity|flip_horiz).

Strategy (verified exact in graph_np.py):
  - Einsum projects one-hot input to 1D presence profiles (any/red/cyan) along rows & cols.
  - Detect orientation via NT = (cyan rows count == 1) -> horizontal vs transposed.
  - Pre-select profiles into a canonical staircase frame, build ONE gather-shear
    staircase (canon [20,20]) + its transpose; select by NT.
  - Red line via outer-Min of red row/col indicators; outside sentinel as 1D
    broadcasts fed into Max. Label padded to 30x30, Equal'd to channel ids.
Cyan is encoded as label value 1 (channel_ids maps channel 8 -> 1).
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

P, Q, K = 30, 20, 5
OUT = Path(__file__).resolve().parent / "cand.onnx"


class B:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def init(self, name, arr):
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def node(self, op, ins, prefix, **attrs):
        self.n += 1
        out = f"{prefix}_{self.n}"
        self.nodes.append(helper.make_node(op, ins, [out], name=out, **attrs))
        return out


def main():
    b = B()

    # --- constants ---
    sel = np.zeros((3, 10), np.float32)
    sel[0, :] = 1.0
    sel[1, 2] = 1.0
    sel[2, 8] = 1.0
    b.init("sel", sel)
    b.init("ones30", np.ones(P, np.float32))
    b.init("zeroF", np.array(0.0, np.float32))

    b.init("axis0", np.array(0, np.int64))
    b.init("one_i32", np.array(1, np.int32))
    b.init("zero_u8", np.array(0, np.uint8))
    b.init("one_u8", np.array(1, np.uint8))
    b.init("two_u8", np.array(2, np.uint8))
    b.init("sent_u8", np.array(255, np.uint8))

    # profile slice bounds over [1,3,20] -> channel s, crop width 20
    b.init("ax12", np.array([1, 2], np.int64))
    for s in range(3):
        b.init(f"st{s}", np.array([s, 0], np.int64))
        b.init(f"en{s}", np.array([s + 1, Q], np.int64))
    b.init("c0_1", np.array([0], np.int64))   # first-elem slice
    b.init("c1_1", np.array([1], np.int64))
    b.init("ax0_1", np.array([0], np.int64))

    b.init("sh20", np.array([Q], np.int64))
    b.init("sh20_1", np.array([Q, 1], np.int64))
    b.init("sh1_20", np.array([1, Q], np.int64))

    # displacement index banks as int32 constants (K rows, length 20 each).
    sig = np.arange(Q)[None, :]
    dd = np.arange(K)[:, None]
    b.init("idxM", np.clip(sig - dd, 0, Q - 1).astype(np.int32))  # [K,20] shift right
    b.init("idxP", np.clip(sig + dd, 0, Q - 1).astype(np.int32))  # [K,20] shift left

    b.init("padsC", np.array([0, 0, P - Q, P - Q], np.int64))   # pad 20x20 -> 30x30
    ch_ids = np.full(10, 254, np.uint8)
    ch_ids[0] = 0
    ch_ids[2] = 2
    ch_ids[8] = 1
    b.init("chan", ch_ids.reshape(1, 10, 1, 1))

    # --- projections ---
    b.node("Einsum", ["input", "sel", "ones30"], "rowp", equation="bchw,sc,w->bsh")
    b.node("Einsum", ["input", "sel", "ones30"], "colp", equation="bchw,sc,h->bsw")
    rowp = b.nodes[-2].output[0]
    colp = b.nodes[-1].output[0]
    presRow = b.node("Greater", [rowp, "zeroF"], "presRow")  # [1,3,30] bool
    presCol = b.node("Greater", [colp, "zeroF"], "presCol")

    def prof(pres, s, nm):
        """Channel s of presence, cropped to 20 -> [20] bool."""
        sl = b.node("Slice", [pres, f"st{s}", f"en{s}", "ax12"], nm + "s")  # [1,1,20]
        return b.node("Reshape", [sl, "sh20"], nm)                          # [20]

    gpR = prof(presRow, 0, "gpR")
    rpR = prof(presRow, 1, "rpR")
    cpR = prof(presRow, 2, "cpR")
    gpC = prof(presCol, 0, "gpC")
    rpC = prof(presCol, 1, "rpC")
    cpC = prof(presCol, 2, "cpC")

    # NT = cyan-rows count == 1  (horizontal case)
    cpRi = b.node("Cast", [cpR], "cpRi", to=TensorProto.INT32)
    cyCnt = b.node("ReduceSum", [cpRi, "ax0_1"], "cyCnt", keepdims=0)
    NT = b.node("Equal", [cyCnt, "one_i32"], "NT")   # scalar bool

    # first elements as u8 [1] (for u8-branch Where), then compared >0 to bool later
    def first_u8(v, nm):
        s = b.node("Slice", [v, "c0_1", "c1_1", "ax0_1"], nm + "b")  # [1] bool
        return b.node("Cast", [s], nm, to=TensorProto.UINT8)         # [1] u8
    cpR0 = first_u8(cpR, "cpR0")
    cpC0 = first_u8(cpC, "cpC0")
    rpR0 = first_u8(rpR, "rpR0")
    rpC0 = first_u8(rpC, "rpC0")

    # --- pre-select profiles into canonical staircase frame ---
    # branches as int32 / u8 (ORT DISABLE_ALL has no bool-branch Where).
    rpRi = b.node("Cast", [rpR], "rpRi", to=TensorProto.INT32)
    rpCi = b.node("Cast", [rpC], "rpCi", to=TensorProto.INT32)
    cum_prof = b.node("Where", [NT, rpRi, rpCi], "cum_prof")            # [20] i32
    cum_side_u8 = b.node("Where", [NT, cpR0, cpC0], "cum_side_u8")      # [1] u8
    cum_side = b.node("Greater", [cum_side_u8, "zero_u8"], "cum_side")  # [1] bool
    # seed: cyan present along seed axis, encoded as u8 1.
    cpC1 = b.node("Where", [cpC, "one_u8", "zero_u8"], "cpC1")
    cpR1 = b.node("Where", [cpR, "one_u8", "zero_u8"], "cpR1")
    srcv = b.node("Where", [NT, cpC1, cpR1], "srcv")                   # [20] u8 (cyan=1)
    shear_u8 = b.node("Where", [NT, rpC0, rpR0], "shear_u8")           # [1] u8
    shear_side = b.node("Greater", [shear_u8, "zero_u8"], "shear_side")  # [1] bool

    cumf = b.node("CumSum", [cum_prof, "axis0"], "cumf")               # [20] i32
    cumr = b.node("CumSum", [cum_prof, "axis0"], "cumr", reverse=1)
    shift = b.node("Where", [cum_side, cumf, cumr], "shift")           # [20] i32

    # displacement banks via Gather with index constants (lowest intermediate memory).
    pos = b.node("Gather", [srcv, "idxM"], "pos", axis=0)            # [K,20]
    neg = b.node("Gather", [srcv, "idxP"], "neg", axis=0)
    bank = b.node("Where", [shear_side, pos, neg], "bank")            # [K,20]
    canon = b.node("Gather", [bank, shift], "canon", axis=0)          # [20,20]
    canon_t = b.node("Transpose", [canon], "canon_t", perm=[1, 0])
    cyan_pat = b.node("Where", [NT, canon, canon_t], "cyan_pat")       # [20,20]

    # red line value (2) via outer-min
    rpR2 = b.node("Where", [rpR, "two_u8", "zero_u8"], "rpR2")
    rpR2c = b.node("Reshape", [rpR2, "sh20_1"], "rpR2c")              # [20,1]
    rpC2 = b.node("Where", [rpC, "two_u8", "zero_u8"], "rpC2")
    rpC2r = b.node("Reshape", [rpC2, "sh1_20"], "rpC2r")             # [1,20]
    red_val = b.node("Min", [rpR2c, rpC2r], "red_val")               # [20,20]

    # outside sentinel as 1D broadcasts
    rowOut1 = b.node("Where", [gpR, "zero_u8", "sent_u8"], "rowOut1")
    rowOut = b.node("Reshape", [rowOut1, "sh20_1"], "rowOut")        # [20,1]
    colOut1 = b.node("Where", [gpC, "zero_u8", "sent_u8"], "colOut1")
    colOut = b.node("Reshape", [colOut1, "sh1_20"], "colOut")       # [1,20]

    color_idx = b.node("Max", [cyan_pat, red_val, rowOut, colOut], "color_idx")  # [20,20]
    label = b.node("Pad", [color_idx, "padsC", "sent_u8"], "label", mode="constant")  # [30,30]
    b.node("Equal", [label, "chan"], "output")
    b.nodes[-1].output[0] = "output"

    graph = helper.make_graph(
        b.nodes, "task382",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, P, P])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, P, P])],
        b.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
