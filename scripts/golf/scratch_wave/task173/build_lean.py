#!/usr/bin/env python3
"""Lean ONNX for task173: label-grid centric, minimal 10-channel tensors.

Validated algorithm in lean_mirror.py. Spatial intermediates are kept at
[1,1,30,30] except 4 grouped-conv armhits (one per shape). LUTs are learned as
tiny [1,10,1,1] vectors and applied via per-color weighted reductions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "scripts/golf/scratch_wave/task173/cand.onnx"

ARM_OFFSETS = {
    0: [(0, 0), (0, 2), (2, 0), (2, 2)],
    1: [(0, 1), (1, 0), (1, 2), (2, 1)],
    2: [(1, 0), (1, 2)],
    3: [(0, 1), (2, 1)],
}
NA = {s: len(ARM_OFFSETS[s]) for s in range(4)}


def armkern(s):
    k = np.zeros((3, 3), np.float32)
    for (r, c) in ARM_OFFSETS[s]:
        k[r, c] = 1.0
    return k


class B:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def name(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def init(self, name, arr):
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def node(self, op, ins, p, **attrs):
        o = self.name(p)
        self.nodes.append(helper.make_node(op, ins, [o], name=o, **attrs))
        return o


def main():
    b = B()
    f32 = TensorProto.FLOAT
    one = b.init("one", np.array(1.0, np.float32))
    ax23 = b.init("ax23", np.array([2, 3], np.int64))
    ax1 = b.init("ax1", np.array([1], np.int64))
    colorvec = b.init("colorvec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))

    # decode label grid
    wdec = b.init("wdec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    L = b.node("Conv", ["input", wdec], "L")  # [1,1,30,30]

    # nz, nzbox
    ch0 = b.node("Slice", ["input",
                           b.init("s0a", np.array([0], np.int64)),
                           b.init("s0b", np.array([1], np.int64)),
                           b.init("s0c", np.array([1], np.int64))], "ch0")
    nz = b.node("Sub", [one, ch0], "nz")  # [1,1,30,30]
    boxw = b.init("boxw", np.ones((1, 1, 3, 3), np.float32))
    nzbox = b.node("Conv", [nz, boxw], "nzbox", pads=[1, 1, 1, 1], kernel_shape=[3, 3])

    # per-shape grouped armhit [1,10,30,30]
    armhit = {}
    for s in range(4):
        w = np.broadcast_to(armkern(s).reshape(1, 1, 3, 3), (10, 1, 3, 3)).copy()
        ws = b.init(f"armw{s}", w)
        armhit[s] = b.node("Conv", ["input", ws], f"armhit{s}", group=10,
                           pads=[1, 1, 1, 1], kernel_shape=[3, 3])

    # ---- learning: per (shape s, arm color a) full-copy presence ----
    # full_s [1,10,30,30] f32 = (armhit_s==nA) & (nz>0) & (nzbox==nA+1), masked ch0=0
    nzch = np.ones((1, 10, 1, 1), np.float32); nzch[0, 0, 0, 0] = 0.0
    nonzero_ch = b.init("nonzero_ch", nzch)

    full = {}
    for s in range(4):
        nA = NA[s]
        eqarm = b.node("Equal", [armhit[s], b.init(f"na{s}", np.array(float(nA), np.float32))], f"eqarm{s}")
        nz_b = b.node("Equal", [nz, one], f"nzb{s}")
        box_b = b.node("Equal", [nzbox, b.init(f"box{s}", np.array(float(nA + 1), np.float32))], f"boxb{s}")
        t1 = b.node("And", [eqarm, nz_b], f"and1_{s}")
        fb = b.node("And", [t1, box_b], f"fullb{s}")
        ff = b.node("Cast", [fb], f"fullf{s}", to=f32)
        full[s] = b.node("Mul", [ff, nonzero_ch], f"fullm{s}")  # [1,10,30,30]

    # arm_shape_s[a] = max spatial full_s[a]  -> [1,10,1,1]
    arm_shape = {}
    for s in range(4):
        arm_shape[s] = b.node("ReduceMax", [full[s], ax23], f"armsh{s}", keepdims=1)  # [1,10,1,1]

    # partner center color per arm color a:
    #   c2p[a] = sum over center color c of c * (anchor of (a) has center c)
    # center color at a full anchor = L. So partner_val_s[a] = max spatial(full_s[a]*L)
    # but multiple anchors share same center color -> max gives that color.
    Lm = L  # [1,1,30,30]
    partner_terms = []
    for s in range(4):
        prod = b.node("Mul", [full[s], Lm], f"pprod{s}")  # broadcast [1,10,30,30]
        red = b.node("ReduceMax", [prod, ax23], f"pred{s}", keepdims=1)  # [1,10,1,1]
        partner_terms.append(red)
    c2p = partner_terms[0]
    for t in partner_terms[1:]:
        c2p = b.node("Max", [c2p, t], "c2p")  # [1,10,1,1] partner center color per arm color a

    # validity: arm color valid if it has any shape (arm_shape) AND partner>0
    arm_any = arm_shape[0]
    for s in range(1, 4):
        arm_any = b.node("Max", [arm_any, arm_shape[s]], "arm_any")  # [1,10,1,1] is-arm
    # gate arm_shape by partner>0 (avoid self/degenerate)
    has_p = b.node("Greater", [c2p, b.init("half", np.array(0.5, np.float32))], "has_p")
    has_pf = b.node("Cast", [has_p], "has_pf", to=f32)
    for s in range(4):
        arm_shape[s] = b.node("Mul", [arm_shape[s], has_pf], f"armsh_g{s}")
    c2p = b.node("Mul", [c2p, has_pf], "c2p_g")

    # ---- arm-pattern anchors ----
    # For each shape s: anchor where armhit_s[a]==nA and a is arm color of shape s.
    # arm color grid_s = max over a of ( (armhit_s[a]==nA) * arm_shape[s][a] ) * a
    # center color grid_s = same positions -> c2p[a].
    arm_a_terms = []   # arm color grids
    arm_cc_terms = []  # center color grids (from arm anchors)
    arm_mask_terms = []
    for s in range(4):
        nA = NA[s]
        eqf = b.node("Cast", [b.node("Equal", [armhit[s], b.init(f"na2_{s}", np.array(float(nA), np.float32))], f"eqf{s}")], f"eqff{s}", to=f32)  # [1,10,30,30]
        sel = b.node("Mul", [eqf, arm_shape[s]], f"sel{s}")  # only channels that are arm of shape s
        # arm color contribution: sel * color index -> reducemax over channels
        a_col = b.node("Mul", [sel, colorvec], f"acol{s}")
        a_red = b.node("ReduceMax", [a_col, ax1], f"ared{s}", keepdims=1)  # [1,1,30,30] arm color
        arm_a_terms.append(a_red)
        # center color contribution: sel * c2p -> reducemax over channels
        cc_col = b.node("Mul", [sel, c2p], f"cccol{s}")
        cc_red = b.node("ReduceMax", [cc_col, ax1], f"ccred{s}", keepdims=1)  # [1,1,30,30]
        arm_cc_terms.append(cc_red)
        # mask: any selected channel
        msk = b.node("ReduceMax", [sel, ax1], f"amask{s}", keepdims=1)  # [1,1,30,30]
        arm_mask_terms.append(msk)
    arm_a = arm_a_terms[0]; arm_cc = arm_cc_terms[0]; arm_mask = arm_mask_terms[0]
    for t in arm_a_terms[1:]:
        arm_a = b.node("Max", [arm_a, t], "arm_a")
    for t in arm_cc_terms[1:]:
        arm_cc = b.node("Max", [arm_cc, t], "arm_cc")
    for t in arm_mask_terms[1:]:
        arm_mask = b.node("Max", [arm_mask, t], "arm_mask")
    arm_mask_b = b.node("Greater", [arm_mask, b.init("half2", np.array(0.5, np.float32))], "arm_mask_b")
    # shape grid for arm anchors: arm_shape_s -> reduce. shape_s = max over channels of sel*s
    arm_s_terms = []
    for s in range(4):
        nA = NA[s]
        eqf = b.node("Cast", [b.node("Equal", [armhit[s], b.init(f"na3_{s}", np.array(float(nA), np.float32))], f"eqf3_{s}")], f"eqff3_{s}", to=f32)
        sel = b.node("Mul", [eqf, arm_shape[s]], f"sel3_{s}")
        ssel = b.node("Mul", [sel, b.init(f"sval{s}", np.full((1, 10, 1, 1), float(s + 1), np.float32))], f"ssel{s}")  # use s+1 to distinguish from 0
        sred = b.node("ReduceMax", [ssel, ax1], f"sred{s}", keepdims=1)
        arm_s_terms.append(sred)
    arm_s = arm_s_terms[0]
    for t in arm_s_terms[1:]:
        arm_s = b.node("Max", [arm_s, t], "arm_s")  # [1,1,30,30], holds s+1 (1..4), 0 if none

    # ---- center-based anchors (ms 0/1): pixel color is a center color ----
    # cc2a[center color] = arm color, cc2s[center color] = shape. Build LUTs [1,10,1,1].
    # center color c is a center iff some arm color a has c2p[a]==c. Then cc2a[c]=a,
    # cc2s[c]=shape(a). Build by scatter over arm colors.
    # cc2a[c]: for each arm color a, place a at index c2p[a]. Do via: for c in 0..9,
    # cc2a[c] = sum_a a * (c2p[a]==c). We'll compute per center color c with compares.
    # Build c2p as length-10 vector; compute cc2a and cc2s as [1,10,1,1].
    # shape-of-arm-color value: arm_s_color[a] = sum_s (s+1)*arm_shape[s][a]
    armshape_val = None
    for s in range(4):
        term = b.node("Mul", [arm_shape[s], b.init(f"sv2_{s}", np.full((1, 10, 1, 1), float(s + 1), np.float32))], f"asv2_{s}")
        armshape_val = term if armshape_val is None else b.node("Add", [armshape_val, term], "armshape_val")
    # cc2a[c], cc2s[c] via per-center compares
    cc2a_cols = []
    cc2s_cols = []
    for c in range(10):
        eqp = b.node("Cast", [b.node("Equal", [c2p, b.init(f"cval{c}", np.array(float(c), np.float32))], f"eqp{c}")], f"eqpf{c}", to=f32)  # [1,10,1,1] 1 where c2p[a]==c
        # exclude c==0
        if c == 0:
            cc2a_cols.append(b.init(f"z_{c}", np.zeros((1, 1, 1, 1), np.float32)))
            cc2s_cols.append(b.init(f"z2_{c}", np.zeros((1, 1, 1, 1), np.float32)))
            continue
        a_contrib = b.node("Mul", [eqp, colorvec], f"acontrib{c}")  # arm color values
        a_red = b.node("ReduceMax", [a_contrib, ax1], f"ccared{c}", keepdims=1)  # [1,1,1,1]
        cc2a_cols.append(a_red)
        s_contrib = b.node("Mul", [eqp, armshape_val], f"scontrib{c}")
        s_red = b.node("ReduceMax", [s_contrib, ax1], f"ccsred{c}", keepdims=1)
        cc2s_cols.append(s_red)
    cc2a = b.node("Concat", cc2a_cols, "cc2a", axis=1)  # [1,10,1,1]
    cc2s = b.node("Concat", cc2s_cols, "cc2s", axis=1)  # [1,10,1,1] holds s+1

    # map L (center color) through cc2a/cc2s. Use one-hot of L = input channels!
    # cen_arm_grid = sum_c input[c]*cc2a[c]  -> reducesum? use Mul+ReduceSum over ch.
    cen_arm = b.node("ReduceSum", [b.node("Mul", ["input", cc2a], "cenarm_m"), ax1], "cen_arm", keepdims=1)  # [1,1,30,30]
    cen_s = b.node("ReduceSum", [b.node("Mul", ["input", cc2s], "cens_m"), ax1], "cen_s", keepdims=1)  # holds s+1
    cen_cc = b.node("ReduceSum", [b.node("Mul", ["input", b.node("Mul", [colorvec, b.node("Cast", [b.node("Greater", [cc2s, b.init("h3", np.array(0.5, np.float32))], "ccg")], "ccgf", to=f32)], "cenmask")], "cencc_m"), ax1], "cen_cc", keepdims=1)  # center color where it IS a center
    cen_mask_b = b.node("Greater", [cen_s, b.init("h4", np.array(0.5, np.float32))], "cen_mask_b")

    # ---- combine anchors: prefer arm-based ----
    arm_mask_f = b.node("Cast", [arm_mask_b], "arm_mask_f", to=f32)
    final_a = b.node("Where", [arm_mask_b, arm_a, cen_arm], "final_a")
    final_s = b.node("Where", [arm_mask_b, arm_s, cen_s], "final_s")  # s+1
    final_cc = b.node("Where", [arm_mask_b, arm_cc, cen_cc], "final_cc")
    anchor_b = b.node("Or", [arm_mask_b, cen_mask_b], "anchor_b")  # [1,1,30,30]
    anchor_f = b.node("Cast", [anchor_b], "anchor_f", to=f32)

    # ---- paint ----
    # For each shape s, anchor positions with final_s==s+1 -> stamp.
    # arm paint: dilate (anchor_f * (final_s==s+1) ) by ARMK[s], weight by final_a.
    # We need per-pixel arm color. Build anchor_a_grid = anchor_f * final_a (arm color
    # at anchor). Dilate per shape, but color must propagate. Trick: per shape,
    # dilate the masked arm-color grid and the masked anchor grid; arm color at a
    # painted cell = dilated_color / dilated_count? Overlap-free so count is 1 per
    # painted cell (within one shape). Use: paint_arm_s = C(anchorcolor_s, ARMK[s]).
    paint_arm = None
    for s in range(4):
        sel_s = b.node("Cast", [b.node("Equal", [final_s, b.init(f"sp{s}", np.array(float(s + 1), np.float32))], f"selse{s}")], f"selsf{s}", to=f32)  # [1,1,30,30]
        anc_s = b.node("Mul", [anchor_f, sel_s], f"ancs{s}")  # anchor mask for shape s
        anc_col_s = b.node("Mul", [anc_s, final_a], f"anccol{s}")  # arm color at anchor
        kern = b.init(f"paintk{s}", armkern(s).reshape(1, 1, 3, 3))
        pa = b.node("Conv", [anc_col_s, kern], f"paintconv{s}", pads=[1, 1, 1, 1], kernel_shape=[3, 3])
        paint_arm = pa if paint_arm is None else b.node("Add", [paint_arm, pa], "paint_arm")  # [1,1,30,30] arm color sum (non-overlap)
    # center paint: at anchor positions, center color = final_cc.
    paint_cc = b.node("Mul", [anchor_f, final_cc], "paint_cc")  # [1,1,30,30]

    # combine: out label = max(input label, paint_arm, paint_cc). Non-overlap so
    # painted arm cells get arm color, centers get center color. But input cells
    # already carry their colors; painting recovers blacked cells. Use Max of label.
    out_lab = b.node("Max", [L, paint_arm], "out_lab1")
    out_lab = b.node("Max", [out_lab, paint_cc], "out_lab")  # [1,1,30,30]

    # ---- to one-hot output with in-grid mask ----
    # output[c] = (out_lab == c) AND in-grid. in-grid = nz OR (L==0 & input has ch0=1)
    # in-grid = ch0 + nz (==1 inside grid). Actually sum of all channels of input.
    ingrid = b.node("ReduceSum", [b.node("Mul", ["input", b.init("ones10", np.ones((1, 10, 1, 1), np.float32))], "ig_m"), ax1], "ingrid", keepdims=1)  # [1,1,30,30] 1 inside grid
    ingrid_b = b.node("Greater", [ingrid, b.init("h5", np.array(0.5, np.float32))], "ingrid_b")
    # build one-hot via Equal(out_lab, arange) then AND ingrid
    arange = b.init("arange", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    eqc = b.node("Equal", [out_lab, arange], "eqc")  # [1,10,30,30] bool (broadcast)
    out = b.node("And", [eqc, ingrid_b], "out_and")  # [1,10,30,30] bool

    graph = helper.make_graph(
        b.nodes, "task173",
        [helper.make_tensor_value_info("input", f32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        b.inits,
    )
    graph.node[-1].output[0] = "output"
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
