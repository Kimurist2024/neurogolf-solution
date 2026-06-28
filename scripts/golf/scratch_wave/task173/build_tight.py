#!/usr/bin/env python3
"""Tight ONNX for task173. Minimize total tensor bytes (sum, not max).

Key budget facts (from playbook):
 - decode conv f32 [1,1,30,30] = 3600 B; label uint8 = 900 B.
 - ReduceMax/Min are f32/int32 only; Conv needs float.
 - output-as-Equal makes the final one-hot free (output excluded).

Algorithm: validated in sc_mirror.py, with the variance check dropped (proven
exact) so we avoid L2/sumL2 convs.

Plan to keep tensor count tiny:
 - decode L (f32). nz from channel-0.
 - Per shape: 2 convs (cntNZ, sumL). Detect uniform arm via cntNZ==nA and integer
   arm color. Combine the 4 shapes' results into compact grids.
 - Build LUTs c2p/c2s via a SINGLE 10-channel one-hot reduction.
 - Apply LUTs via Gather. Paint via 4 convs. Output via Equal.
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
F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
I32 = TensorProto.INT32


def armkern1(s):
    k = np.zeros((1, 1, 3, 3), np.float32)
    for (r, c) in ARM_OFFSETS[s]:
        k[0, 0, r, c] = 1.0
    return k


class B:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0
        self._c = {}

    def nm(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def init(self, name, arr):
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def k(self, val, dt=np.float32):
        key = (float(val), str(dt))
        if key not in self._c:
            self._c[key] = self.init(self.nm("c"), np.array(val, dt))
        return self._c[key]

    def node(self, op, ins, p, **a):
        o = self.nm(p)
        self.nodes.append(helper.make_node(op, ins, [o], name=o, **a))
        return o


def main():
    b = B()
    conv3 = dict(pads=[1, 1, 1, 1], kernel_shape=[3, 3])
    ax23 = b.init("ax23", np.array([2, 3], np.int64))
    ax1 = b.init("ax1", np.array([1], np.int64))
    arange = b.init("arange", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))

    # decode L
    wdec = b.init("wdec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    L = b.node("Conv", ["input", wdec], "L")  # f32 [1,1,30,30]

    # in-grid + nz
    ig = b.node("ReduceSum", ["input", ax1], "ig", keepdims=1)  # [1,1,30,30] 1 inside grid
    ch0 = b.node("Slice", ["input",
                           b.init("z0", np.array([0], np.int64)),
                           b.init("z1", np.array([1], np.int64)),
                           b.init("z2", np.array([1], np.int64))], "ch0")
    # nz = in-grid AND nonzero color = ingrid - ch0 (out-of-grid cells have ch0=0
    # AND ingrid=0, so nz=0; in-grid bg has ingrid=1,ch0=1 -> 0; in-grid color ->1)
    nz = b.node("Sub", [ig, ch0], "nz")  # f32 {0,1}
    boxw = b.init("boxw", np.ones((1, 1, 3, 3), np.float32))
    nzbox = b.node("Conv", [nz, boxw], "nzbox", **conv3)

    # ---- per-shape arm detection: cntNZ, sumL via convs ----
    # combine arm-color grid and shape grid (over shapes) for both arm anchors and
    # full-copy anchors.
    arm_color = {}     # f32 grid: arm color where uniform-of-shape-s (else 0)
    arm_color_int = {}  # int32 grid: rounded arm color
    arm_uni = {}       # f32 {0,1}
    full = {}          # f32 {0,1} full-copy anchor of shape s
    for s in range(4):
        nA = NA[s]
        ker = b.init(f"ak{s}", armkern1(s))
        cnt = b.node("Conv", [nz, ker], f"cnt{s}", **conv3)
        sm = b.node("Conv", [L, ker], f"sm{s}", **conv3)
        eqn = b.node("Equal", [cnt, b.k(float(nA))], f"eqn{s}")  # cnt==nA
        ac = b.node("Mul", [sm, b.k(1.0 / nA)], f"ac{s}")  # sum/nA
        # integer check: round(ac)==ac
        aci = b.node("Cast", [b.node("Add", [ac, b.k(0.5)], f"acr{s}")], f"aci{s}", to=I32)
        arm_color_int[s] = aci  # reuse for Gather below
        acf = b.node("Cast", [aci, ], f"acf{s}", to=F32)
        int_ok = b.node("Equal", [acf, ac], f"into{s}")
        uni_b = b.node("And", [eqn, int_ok], f"unib{s}")
        uni = b.node("Cast", [uni_b, ], f"uni{s}", to=F32)
        arm_uni[s] = uni
        arm_color[s] = b.node("Mul", [acf, uni], f"acm{s}")  # arm color (int) where uniform
        # full copy: uni & nz & nzbox==nA+1
        nzb = b.node("Greater", [nz, b.k(0.5)], f"nzg{s}")
        boxe = b.node("Equal", [nzbox, b.k(float(nA + 1))], f"bxe{s}")
        t = b.node("And", [uni_b, nzb], f"ft{s}")
        fb = b.node("And", [t, boxe], f"fb{s}")
        full[s] = b.node("Cast", [fb, ], f"ff{s}", to=F32)

    # ---- LUT build via single 10-channel one-hot reduction ----
    # armcolor_full grid: arm color at full-copy anchors (combine shapes)
    # shape_full grid: shape (s+1) at full-copy anchors
    acfull = None; sfull = None; ccfull = L  # center color at anchor = L
    for s in range(4):
        ac_s = b.node("Mul", [arm_color[s], full[s]], f"acfs{s}")
        sv_s = b.node("Mul", [full[s], b.k(float(s + 1))], f"sfs{s}")
        acfull = ac_s if acfull is None else b.node("Max", [acfull, ac_s], f"acfu{s}")
        sfull = sv_s if sfull is None else b.node("Max", [sfull, sv_s], f"sfu{s}")
    # Build LUTs via ScatterND (avoids the [1,10,30,30] one-hot, ~108KB saved).
    # indices = arm color at full anchors (int), updates = center color (L) / shape.
    # Non-anchor cells have acfull=0 -> scatter into index 0 (masked out later).
    shp900_1 = b.init("shp900_1", np.array([900, 1], np.int64))
    shp900 = b.init("shp900", np.array([900], np.int64))
    shp10 = b.init("shp10", np.array([10], np.int64))
    acfull_i = b.node("Cast", [b.node("Add", [acfull, b.k(0.5)], "acfr")], "acfull_i", to=I64)
    idx_flat = b.node("Reshape", [acfull_i, shp900_1], "idx_flat")  # [900,1]
    L_flat = b.node("Reshape", [L, shp900], "L_flat")  # [900]
    sfull_flat = b.node("Reshape", [sfull, shp900], "sfull_flat")  # [900]
    zeros10 = b.init("zeros10", np.zeros(10, np.float32))
    c2p_raw = b.node("ScatterND", [zeros10, idx_flat, L_flat], "c2p_raw")  # [10]
    c2s_raw = b.node("ScatterND", [zeros10, idx_flat, sfull_flat], "c2s_raw")  # [10] s+1
    # zero index 0 (arm color 0 invalid)
    zmask10 = np.ones(10, np.float32); zmask10[0] = 0.0
    zmask = b.init("zmask10", zmask10)
    c2p = b.node("Mul", [c2p_raw, zmask], "c2p")
    c2s = b.node("Mul", [c2s_raw, zmask], "c2s")
    # column form for inverse-LUT build below
    c2p_v = b.node("Reshape", [c2p, b.init("shp_1_10_1_1", np.array([1, 10, 1, 1], np.int64))], "c2p_v")
    c2s_v = b.node("Reshape", [c2s, b.init("shp_1_10_1_1b", np.array([1, 10, 1, 1], np.int64))], "c2s_v")

    # center LUTs: cc2a[c], cc2s[c] = inverse of c2p. Build [10] via one-hot of c2p.
    # for center color c: arm color a with c2p[a]==c. cc2a[c]=a, cc2s[c]=c2s[a].
    # one-hot c2p over center colors: ohp[a,c] = (c2p[a]==c). Build [1,10,10].
    c2p_col = b.node("Reshape", [c2p_v, b.init("shp10c", np.array([1, 10, 1, 1], np.int64))], "c2pcol")  # [1,10,1,1]
    arange_c = b.init("arange_c", np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1))
    ohp = b.node("Cast", [b.node("Equal", [c2p_col, arange_c], "ohpeq")], "ohp_raw", to=F32)  # [1,10,10,1]
    # mask out a==0 (axis1) and c==0 (axis2): only valid arm/center colors 1..9.
    acmask = np.ones((1, 10, 10, 1), np.float32)
    acmask[0, 0, :, 0] = 0.0
    acmask[0, :, 0, 0] = 0.0
    ohp = b.node("Mul", [ohp, b.init("acmask", acmask)], "ohp")
    # cc2a[c] = sum_a a*ohp[a,c] : weight a along axis1
    avec = b.init("avec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    cc2a_v = b.node("ReduceSum", [b.node("Mul", [ohp, avec], "ohpa"), ax1], "cc2a_v", keepdims=1)  # [1,1,10,1]
    # cc2s[c] = sum_a c2s[a]*ohp[a,c]
    c2s_col = b.node("Reshape", [c2s_v, b.init("shp10c2", np.array([1, 10, 1, 1], np.int64))], "c2scol")
    cc2s_v = b.node("ReduceSum", [b.node("Mul", [ohp, c2s_col], "ohps"), ax1], "cc2s_v", keepdims=1)  # [1,1,10,1] s+1
    cc2a = b.node("Reshape", [cc2a_v, shp10], "cc2a")
    cc2s = b.node("Reshape", [cc2s_v, shp10], "cc2s")  # s+1; 0 means not a center color

    # ---- arm-pattern anchors ----
    L_int = b.node("Cast", [b.node("Add", [L, b.k(0.5)], "Lr")], "L_int", to=I32)
    final_a = None; final_s = None; final_cc = None; arm_mask = None
    for s in range(4):
        aci = arm_color_int[s]  # reuse int32 cast from detection
        c2s_here = b.node("Gather", [c2s, aci], f"c2sh{s}", axis=0)  # s+1 for this arm color
        match = b.node("And", [b.node("Equal", [c2s_here, b.k(float(s + 1))], f"mch{s}"),
                               b.node("Greater", [arm_uni[s], b.k(0.5)], f"ung{s}")], f"vld{s}")
        sel = b.node("Cast", [match], f"selc{s}", to=F32)  # [1,1,30,30]
        c2p_here = b.node("Gather", [c2p, aci], f"c2ph{s}", axis=0)
        a_g = b.node("Mul", [sel, arm_color[s]], f"ag{s}")
        cc_g = b.node("Mul", [sel, c2p_here], f"ccg{s}")
        s_g = b.node("Mul", [sel, b.k(float(s + 1))], f"sg{s}")
        final_a = a_g if final_a is None else b.node("Max", [final_a, a_g], f"fa{s}")
        final_cc = cc_g if final_cc is None else b.node("Max", [final_cc, cc_g], f"fcc{s}")
        final_s = s_g if final_s is None else b.node("Max", [final_s, s_g], f"fs{s}")
        arm_mask = sel if arm_mask is None else b.node("Max", [arm_mask, sel], f"am{s}")
    arm_mask_b = b.node("Greater", [arm_mask, b.k(0.5)], "ammb")

    # ---- center-based anchors ----
    cen_arm = b.node("Gather", [cc2a, L_int], "cenarm", axis=0)  # arm color
    cen_s_l = b.node("Gather", [cc2s, L_int], "censl", axis=0)   # s+1 (0 if not center)
    cen_mask_b = b.node("Greater", [cen_s_l, b.k(0.5)], "cmb")
    cen_cc = b.node("Mul", [L, b.node("Cast", [cen_mask_b], "cmf", to=F32)], "cencc")

    # combine (prefer arm)
    fa = b.node("Where", [arm_mask_b, final_a, cen_arm], "FA")
    fs = b.node("Where", [arm_mask_b, final_s, cen_s_l], "FS")  # s+1
    fcc = b.node("Where", [arm_mask_b, final_cc, cen_cc], "FCC")
    anchor_b = b.node("Or", [arm_mask_b, cen_mask_b], "anchorb")
    anchor_f = b.node("Cast", [anchor_b], "anchorf", to=F32)

    # ---- paint ----
    paint = None
    for s in range(4):
        sel_s = b.node("Cast", [b.node("Equal", [fs, b.k(float(s + 1))], f"pse{s}")], f"psf{s}", to=F32)
        ancc = b.node("Mul", [b.node("Mul", [anchor_f, sel_s], f"pa{s}"), fa], f"pac{s}")
        ker = b.init(f"pk{s}", armkern1(s))
        pc = b.node("Conv", [ancc, ker], f"pcv{s}", **conv3)
        paint = pc if paint is None else b.node("Add", [paint, pc], f"pp{s}")
    paint_cc = b.node("Mul", [anchor_f, fcc], "paintcc")
    out_lab = b.node("Max", [b.node("Max", [L, paint], "ol1"), paint_cc], "out_lab")

    # ---- output ----
    ig_b = b.node("Greater", [ig, b.k(0.5)], "igb")
    eqc = b.node("Equal", [out_lab, arange], "eqc")
    out = b.node("And", [eqc, ig_b], "out_and")

    graph = helper.make_graph(
        b.nodes, "task173",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
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
