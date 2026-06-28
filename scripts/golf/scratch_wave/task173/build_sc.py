#!/usr/bin/env python3
"""Single-channel ONNX for task173. All spatial tensors are [1,1,30,30].

Algorithm validated in sc_mirror.py. LUTs built via per-color single-channel
reductions; applied via Gather on integer label grids.
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


def armkern(s):
    k = np.zeros((1, 1, 3, 3), np.float32)
    for (r, c) in ARM_OFFSETS[s]:
        k[0, 0, r, c] = 1.0
    return k


class B:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0
        self._cache = {}

    def name(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def init(self, name, arr):
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def const(self, val, dtype=np.float32):
        key = (float(val), str(dtype))
        if key in self._cache:
            return self._cache[key]
        nm = self.init(self.name("k"), np.array(val, dtype))
        self._cache[key] = nm
        return nm

    def node(self, op, ins, p, **attrs):
        o = self.name(p)
        self.nodes.append(helper.make_node(op, ins, [o], name=o, **attrs))
        return o


def main():
    b = B()
    conv3 = dict(pads=[1, 1, 1, 1], kernel_shape=[3, 3])

    # decode label grid [1,1,30,30] f32
    wdec = b.init("wdec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    L = b.node("Conv", ["input", wdec], "L")
    L2 = b.node("Mul", [L, L], "L2")

    # nz, nzbox
    ones10 = b.init("ones10", np.ones((1, 10, 1, 1), np.float32))
    ig_m = b.node("Mul", ["input", ones10], "ig_m")
    ingrid = b.node("ReduceSum", [ig_m, b.init("axc", np.array([1], np.int64))], "ingrid", keepdims=1)  # [1,1,30,30]
    nz = ingrid  # inside grid all-channels sum is 1; equals nonzero? No: ch0=1 for bg.
    # Actually nz (nonzero color) = 1 - ch0.
    ch0 = b.node("Slice", ["input",
                           b.init("s0a", np.array([0], np.int64)),
                           b.init("s0b", np.array([1], np.int64)),
                           b.init("s0c", np.array([1], np.int64))], "ch0")
    nz = b.node("Sub", [b.const(1.0), ch0], "nz")
    boxw = b.init("boxw", np.ones((1, 1, 3, 3), np.float32))
    nzbox = b.node("Conv", [nz, boxw], "nzbox", **conv3)

    # per shape: uniform arm detection
    arm_uniform = {}
    arm_color = {}        # f32 grid
    arm_color_int = {}    # int64 grid
    for s in range(4):
        nA = NA[s]
        k = b.init(f"armk{s}", armkern(s))
        cntNZ = b.node("Conv", [nz, k], f"cntNZ{s}", **conv3)
        sumL = b.node("Conv", [L, k], f"sumL{s}", **conv3)
        sumL2 = b.node("Conv", [L2, k], f"sumL2{s}", **conv3)
        eq_cnt = b.node("Equal", [cntNZ, b.const(float(nA))], f"eqcnt{s}")
        # nA*sumL2 == sumL^2
        nasum2 = b.node("Mul", [sumL2, b.const(float(nA))], f"nasum2_{s}")
        sumLsq = b.node("Mul", [sumL, sumL], f"sumLsq{s}")
        eq_var = b.node("Equal", [nasum2, sumLsq], f"eqvar{s}")
        uni = b.node("And", [eq_cnt, eq_var], f"uni{s}")
        uni_f = b.node("Cast", [uni], f"unif{s}", to=F32)
        arm_uniform[s] = uni_f
        ac = b.node("Mul", [sumL, b.const(1.0 / nA)], f"ac{s}")  # arm color where uniform
        ac = b.node("Mul", [ac, uni_f], f"acm{s}")  # zero outside uniform
        arm_color[s] = ac
        arm_color_int[s] = b.node("Cast", [b.node("Add", [ac, b.const(0.5)], f"acr{s}")], f"aci{s}", to=I64)

    # ---- LUT learning ----
    # full_s = uniform & nz>0 & nzbox==nA+1
    full = {}
    for s in range(4):
        nA = NA[s]
        nz_b = b.node("Greater", [nz, b.const(0.5)], f"nzg{s}")
        box_b = b.node("Equal", [nzbox, b.const(float(nA + 1))], f"boxe{s}")
        uni_b = b.node("Greater", [arm_uniform[s], b.const(0.5)], f"unib{s}")
        t = b.node("And", [uni_b, nz_b], f"ft{s}")
        fb = b.node("And", [t, box_b], f"fb{s}")
        full[s] = b.node("Cast", [fb], f"ffull{s}", to=F32)  # [1,1,30,30]

    # Build c2p[a] (partner center color) and c2s[a] (shape) for a in 1..9.
    ax23 = b.init("ax23", np.array([2, 3], np.int64))
    c2p_scalars = [b.init("c2p0", np.zeros((1, 1, 1, 1), np.float32))]  # a=0 -> 0
    c2s_scalars = [b.init("c2s0", np.full((1, 1, 1, 1), -1.0, np.float32))]
    for a in range(1, 10):
        # mask where arm color == a across shapes (with full)
        best_cc = None  # center color
        best_s = None   # shape value (s, encoded as s; -1 if none)
        for s in range(4):
            eqa = b.node("Cast", [b.node("Equal", [arm_color[s], b.const(float(a))], f"eqa{a}_{s}")], f"eqaf{a}_{s}", to=F32)
            m = b.node("Mul", [eqa, full[s]], f"m{a}_{s}")  # 1 where full copy of arm color a, shape s
            # center color there = L. partner contribution = max(m*L)
            ccc = b.node("ReduceMax", [b.node("Mul", [m, L], f"mcc{a}_{s}"), ax23], f"ccr{a}_{s}", keepdims=1)  # [1,1,1,1]
            sval = b.node("ReduceMax", [b.node("Mul", [m, b.const(float(s + 1))], f"msv{a}_{s}"), ax23], f"svr{a}_{s}", keepdims=1)  # s+1 if present else 0
            best_cc = ccc if best_cc is None else b.node("Max", [best_cc, ccc], f"bcc{a}_{s}")
            best_s = sval if best_s is None else b.node("Max", [best_s, sval], f"bsv{a}_{s}")
        c2p_scalars.append(best_cc)  # [1,1,1,1]
        # convert s+1 (1..4, 0 if none) to shape (0..3, -1 if none)
        c2s_val = b.node("Sub", [best_s, b.const(1.0)], f"c2sv{a}")  # -1 if none, else s
        c2s_scalars.append(c2s_val)

    # assemble LUT vectors as [10] along axis 1, then reshape to [10] for Gather.
    c2p_vec = b.node("Concat", c2p_scalars, "c2p_vec", axis=1)  # [1,10,1,1]
    c2s_vec = b.node("Concat", c2s_scalars, "c2s_vec", axis=1)  # [1,10,1,1]
    # reshape to [10]
    shp10 = b.init("shp10", np.array([10], np.int64))
    c2p_flat = b.node("Reshape", [c2p_vec, shp10], "c2p_flat")  # [10]
    c2s_flat = b.node("Reshape", [c2s_vec, shp10], "c2s_flat")  # [10]

    # center LUTs cc2a[c], cc2s[c]: scatter arm color a -> index c2p[a]. Build via
    # per center color c loop: cc2a[c] = max over a of a*(c2p[a]==c).
    cc2a_scalars = [b.init("cc2a0", np.zeros((1, 1, 1, 1), np.float32))]
    cc2s_scalars = [b.init("cc2s0", np.full((1, 1, 1, 1), -1.0, np.float32))]
    # need per-a scalars c2p[a], c2s[a] -> slice from vectors
    for c in range(1, 10):
        best_a = None
        best_s = None
        for a in range(1, 10):
            c2p_a = b.node("Slice", [c2p_vec,
                                     b.init(f"c2psa{c}_{a}", np.array([a], np.int64)),
                                     b.init(f"c2psb{c}_{a}", np.array([a + 1], np.int64)),
                                     b.init(f"c2psc{c}_{a}", np.array([1], np.int64))], f"c2pa{c}_{a}")  # [1,1,1,1]
            eqc = b.node("Cast", [b.node("Equal", [c2p_a, b.const(float(c))], f"cceq{c}_{a}")], f"cceqf{c}_{a}", to=F32)
            acontrib = b.node("Mul", [eqc, b.const(float(a))], f"cca{c}_{a}")
            best_a = acontrib if best_a is None else b.node("Max", [best_a, acontrib], f"cba{c}_{a}")
            c2s_a = b.node("Slice", [c2s_vec,
                                     b.init(f"c2ssa{c}_{a}", np.array([a], np.int64)),
                                     b.init(f"c2ssb{c}_{a}", np.array([a + 1], np.int64)),
                                     b.init(f"c2ssc{c}_{a}", np.array([1], np.int64))], f"c2sa{c}_{a}")
            # shape contribution (s+1) gated by eqc; store max, then subtract 1
            sp1 = b.node("Add", [c2s_a, b.const(1.0)], f"sp1_{c}_{a}")
            scontrib = b.node("Mul", [eqc, sp1], f"ccs{c}_{a}")
            best_s = scontrib if best_s is None else b.node("Max", [best_s, scontrib], f"cbs{c}_{a}")
        cc2a_scalars.append(best_a)
        cc2s_scalars.append(b.node("Sub", [best_s, b.const(1.0)], f"cc2sv{c}"))  # -1 if none

    cc2a_vec = b.node("Concat", cc2a_scalars, "cc2a_vec", axis=1)  # [1,10,1,1]
    cc2s_vec = b.node("Concat", cc2s_scalars, "cc2s_vec", axis=1)
    cc2a_flat = b.node("Reshape", [cc2a_vec, shp10], "cc2a_flat")
    cc2s_flat = b.node("Reshape", [cc2s_vec, shp10], "cc2s_flat")

    # ---- arm-pattern anchors ----
    # For shape s: anchor where uniform AND c2s[arm_color]==s. Look up c2s via Gather.
    L_int = b.node("Cast", [b.node("Add", [L, b.const(0.5)], "Lr"), ], "L_int", to=I64)  # [1,1,30,30]
    arm_a = None; arm_s = None; arm_cc = None; arm_mask = None
    for s in range(4):
        # shape of the uniform arm color at each pixel
        c2s_here = b.node("Gather", [c2s_flat, arm_color_int[s]], f"c2shere{s}", axis=0)  # [1,1,30,30]
        match = b.node("Cast", [b.node("Equal", [c2s_here, b.const(float(s))], f"matche{s}")], f"matchf{s}", to=F32)
        sel = b.node("Mul", [match, arm_uniform[s]], f"selm{s}")  # 1 where valid arm anchor of shape s
        a_grid = b.node("Mul", [sel, arm_color[s]], f"aga{s}")
        # center color = c2p[arm_color]: Gather
        c2p_here = b.node("Gather", [c2p_flat, arm_color_int[s]], f"c2phere{s}", axis=0)
        cc_grid = b.node("Mul", [sel, c2p_here], f"agc{s}")
        s_grid = b.node("Mul", [sel, b.const(float(s + 1))], f"ags{s}")  # s+1
        arm_a = a_grid if arm_a is None else b.node("Max", [arm_a, a_grid], f"arma{s}")
        arm_cc = cc_grid if arm_cc is None else b.node("Max", [arm_cc, cc_grid], f"armcc{s}")
        arm_s = s_grid if arm_s is None else b.node("Max", [arm_s, s_grid], f"arms{s}")
        arm_mask = sel if arm_mask is None else b.node("Max", [arm_mask, sel], f"armm{s}")
    arm_mask_b = b.node("Greater", [arm_mask, b.const(0.5)], "arm_mask_b")  # [1,1,30,30]

    # ---- center-based anchors ----
    cen_arm = b.node("Gather", [cc2a_flat, L_int], "cen_arm", axis=0)  # [1,1,30,30]
    cen_s_lut = b.node("Gather", [cc2s_flat, L_int], "cen_s_lut", axis=0)  # holds shape (0..3) or -1
    cen_mask_b = b.node("Greater", [cen_s_lut, b.const(-0.5)], "cen_mask_b")  # >=0 -> is center
    cen_s = b.node("Add", [cen_s_lut, b.const(1.0)], "cen_s")  # s+1, or 0 if -1
    cen_s = b.node("Mul", [cen_s, b.node("Cast", [cen_mask_b], "cmf", to=F32)], "cen_s2")
    cen_cc = b.node("Mul", [L, b.node("Cast", [cen_mask_b], "cmf2", to=F32)], "cen_cc")  # center color = L

    # combine, prefer arm-based
    final_a = b.node("Where", [arm_mask_b, arm_a, cen_arm], "final_a")
    final_s = b.node("Where", [arm_mask_b, arm_s, cen_s], "final_s")  # s+1
    final_cc = b.node("Where", [arm_mask_b, arm_cc, cen_cc], "final_cc")
    anchor_b = b.node("Or", [arm_mask_b, cen_mask_b], "anchor_b")
    anchor_f = b.node("Cast", [anchor_b], "anchor_f", to=F32)

    # ---- paint ----
    paint_arm = None
    for s in range(4):
        sel_s = b.node("Cast", [b.node("Equal", [final_s, b.const(float(s + 1))], f"pse{s}")], f"psef{s}", to=F32)
        anc_col = b.node("Mul", [b.node("Mul", [anchor_f, sel_s], f"panc{s}"), final_a], f"pancc{s}")
        k = b.init(f"paintk{s}", armkern(s))
        pa = b.node("Conv", [anc_col, k], f"paintconv{s}", **conv3)
        paint_arm = pa if paint_arm is None else b.node("Add", [paint_arm, pa], f"parm{s}")
    paint_cc = b.node("Mul", [anchor_f, final_cc], "paint_cc")

    out_lab = b.node("Max", [L, paint_arm], "outl1")
    out_lab = b.node("Max", [out_lab, paint_cc], "out_lab")

    # ---- to one-hot ----
    ingrid_b = b.node("Greater", [ingrid, b.const(0.5)], "ingrid_b")
    arange = b.init("arange", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    eqc = b.node("Equal", [out_lab, arange], "eqc")  # [1,10,30,30]
    out = b.node("And", [eqc, ingrid_b], "out_and")

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
