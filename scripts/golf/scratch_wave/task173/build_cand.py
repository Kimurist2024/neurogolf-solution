#!/usr/bin/env python3
"""Build a compact ONNX for task173 (ARC 72322fa7, archetype overlaps).

Strategy (validated exact in tensor_solver.py):
  1. Decode one-hot -> label grid L [1,1,30,30] f32.
  2. nz mask. Per shape s (4 fixed arm kernels) compute armhit over all 10
     channels via grouped Conv. Detect full copies -> learn, per arm color,
     its shape and partner center color (tiny [10,*] reductions).
  3. Build per-color anchor maps, dilate arms, paint centers, OR with input.
  4. Emit one-hot output [1,10,30,30] bool.

We keep spatial intermediates in f32 (Conv requires float) but collapse via
ReduceMax over channel groups where possible. Cost is dominated by [1,10,30,30]
f32 tensors (36000 B each), so we minimize their count.
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

    # --- decode one-hot -> label grid f32 [1,1,30,30] ---
    wdec = b.init("wdec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    L = b.node("Conv", ["input", wdec], "L")  # [1,1,30,30] f32

    # nz mask: sum of channels 1..9 = 1 - channel0. Use ReduceSum over c=1..9.
    # Easier: nz = (L>0). Build via channel-0 complement.
    # sum all 10 channels = 1 everywhere inside grid, 0 outside. channel0 = bg.
    # nz = 1 - X[:,0]. Slice channel 0.
    ch0 = b.node("Slice", ["input",
                           b.init("s0a", np.array([0], np.int64)),
                           b.init("s0b", np.array([1], np.int64)),
                           b.init("s0c", np.array([1], np.int64))], "ch0")  # [1,1,30,30]
    one = b.init("one_f", np.array(1.0, np.float32))
    nz = b.node("Sub", [one, ch0], "nz")  # [1,1,30,30] f32 in {0,1} (1 where nonzero)

    # nzbox = box3 conv of nz
    boxw = b.init("boxw", np.ones((1, 1, 3, 3), np.float32))
    nzbox = b.node("Conv", [nz, boxw], "nzbox", pads=[1, 1, 1, 1], kernel_shape=[3, 3])  # [1,1,30,30]

    # --- per-shape armhit over all 10 channels (grouped conv) ---
    # weight [10,1,3,3] = same arm kernel replicated across channels (group=10)
    armhit = {}
    for s in range(4):
        w = np.broadcast_to(armkern(s).reshape(1, 1, 3, 3), (10, 1, 3, 3)).copy()
        ws = b.init(f"armw{s}", w)
        armhit[s] = b.node("Conv", ["input", ws], f"armhit{s}", group=10,
                           pads=[1, 1, 1, 1], kernel_shape=[3, 3])  # [1,10,30,30]

    # --- learn arm_shape[k,s] and arm_partner[k,c] ---
    # full_s [1,10,30,30] = (armhit_s==nA_s) & (nz_center) & (nzbox==nA_s+1)
    # We compute, per shape, full anchors, then reduce over spatial -> arm_shape.
    # Build comparison constants.
    def eq_const(x, val, p):
        c = b.init(f"{p}c", np.array(val, np.float32))
        return b.node("Equal", [x, c], p)  # bool

    arm_shape_cols = []  # [1,10,1,1] per shape, then concat? We'll keep per-color.
    # We need arm_partner[k,c]. Build big but reduce quickly.
    # Compose anchor full masks per shape (bool->f32).
    # channel-0 mask: 0 for background channel, 1 for colors 1..9
    nzch = np.ones((1, 10, 1, 1), np.float32)
    nzch[0, 0, 0, 0] = 0.0
    nonzero_ch = b.init("nonzero_ch", nzch)

    full = {}
    for s in range(4):
        nA = NA[s]
        eqarm = b.node("Equal", [armhit[s], b.init(f"na{s}", np.array(float(nA), np.float32))], f"eqarm{s}")  # bool [1,10,30,30]
        # nz center as bool, broadcast: nz==1
        nz_b = b.node("Equal", [nz, b.init(f"nz1_{s}", np.array(1.0, np.float32))], f"nzb{s}")  # [1,1,30,30] bool
        box_b = b.node("Equal", [nzbox, b.init(f"box{s}", np.array(float(nA + 1), np.float32))], f"boxb{s}")  # [1,1,30,30]
        t1 = b.node("And", [eqarm, nz_b], f"and1_{s}")  # broadcast -> [1,10,30,30]
        fb = b.node("And", [t1, box_b], f"fullb{s}")  # [1,10,30,30] bool
        full_raw = b.node("Cast", [fb], f"fullf{s}", to=TensorProto.FLOAT)  # f32
        # arm color can never be background (channel 0). Zero channel 0.
        full[s] = b.node("Mul", [full_raw, nonzero_ch], f"fullmask{s}")

    # arm_shape[k,s] = max over spatial of full[s][k]  -> [1,10,1,1]
    axHW = b.init("axHW", np.array([2, 3], np.int64))
    arm_shape_s = {}
    for s in range(4):
        arm_shape_s[s] = b.node("ReduceMax", [full[s], axHW], f"armshape{s}", keepdims=1)  # [1,10,1,1]

    # arm_partner[k,c]: for shape s, anchors where center color is c -> full[s][k]*X[c].
    # We need, per arm color k, the partner center color channel c. Build a
    # [1,10(k),10(c)] tensor by, for each c, reducemax over spatial of
    # (sum_s full[s]) * X[c]. Then argmax over c gives partner; but we want to use
    # it to (a) gate validity (partner exists & !=k) and (b) select center color.
    # full_any[k] = max_s full[s][k]
    full_any = full[0]
    for s in range(1, 4):
        full_any = b.node("Max", [full_any, full[s]], "full_any")  # [1,10,30,30] f32 (over k)

    # For partner: reduce over spatial of full_any[k]*X[c] for each c. To get a
    # [1,10,10] map we'd loop c=0..9. Build partner_kc as concat of 10 [1,10,1,1].
    partner_cols = []
    for c in range(10):
        Xc = b.node("Slice", ["input",
                              b.init(f"pcs{c}a", np.array([c], np.int64)),
                              b.init(f"pcs{c}b", np.array([c + 1], np.int64)),
                              b.init(f"pcs{c}c", np.array([1], np.int64))], f"Xc{c}")  # [1,1,30,30]
        prod = b.node("Mul", [full_any, Xc], f"prod{c}")  # [1,10,30,30]
        red = b.node("ReduceMax", [prod, axHW], f"pred{c}", keepdims=1)  # [1,10,1,1]
        partner_cols.append(red)
    # partner_kc [1,10,10,1] : concat along a new axis. We have 10 tensors [1,10,1,1].
    # Concat along axis=2 -> [1,10,10,1]
    partner_kc = b.node("Concat", partner_cols, "partner_kc", axis=2)  # [1,10,10,1]
    # zero out k==c (self). Build mask [1,10,10,1] with 0 on diagonal.
    diagmask = np.ones((1, 10, 10, 1), np.float32)
    for k in range(10):
        diagmask[0, k, k, 0] = 0.0
    diagmask[0, 0, :, 0] = 0.0  # k=0 (background) never an arm color
    diagmask[0, :, 0, 0] = 0.0  # c=0 (background) never a center color
    partner_kc = b.node("Mul", [partner_kc, b.init("diagmask", diagmask)], "partner_kc2")

    # has_partner[k] = max over c of partner_kc[k,c]  -> [1,10,1,1]
    ax2 = b.init("ax2", np.array([2], np.int64))
    has_partner = b.node("ReduceMax", [partner_kc, ax2], "has_partner", keepdims=1)  # [1,10,1,1]

    # valid arm_shape: arm_shape_s[s] gated by has_partner
    arm_shape_valid = {}
    for s in range(4):
        arm_shape_valid[s] = b.node("Mul", [arm_shape_s[s], has_partner], f"asv{s}")  # [1,10,1,1]

    # --- build anchor map per arm color k ---
    # arm_anchor: where k's own shape arm pattern is full. k's shape selected by
    # arm_shape_valid. armhit for k's shape: sum_s arm_shape_valid[s]*(armhit[s]==nA_s).
    # We need (armhit[s]==nA_s) per shape as f32, then gate by arm_shape_valid[s]
    # (broadcast [1,10,1,1] over spatial), then max over s -> arm_anchor [1,10,30,30].
    arm_anchor_terms = []
    for s in range(4):
        eqf = b.node("Cast", [b.node("Equal", [armhit[s], b.init(f"na2_{s}", np.array(float(NA[s]), np.float32))], f"eqf{s}")], f"eqff{s}", to=TensorProto.FLOAT)
        gated = b.node("Mul", [eqf, arm_shape_valid[s]], f"argated{s}")  # [1,10,30,30]
        arm_anchor_terms.append(gated)
    arm_anchor = arm_anchor_terms[0]
    for t in arm_anchor_terms[1:]:
        arm_anchor = b.node("Max", [arm_anchor, t], "arm_anchor")  # [1,10,30,30]

    # cen_anchor[k]: center color present = sum_c partner_kc[k,c]*X[c] (spatial).
    # partner_kc is [1,10,10,1]; we need to map to spatial. For each c, weight X[c]
    # by partner_kc[:, :, c, :] (a [1,10,1,1]). Sum over c.
    cen_anchor = None
    for c in range(10):
        # slice partner weight for center color c: partner_kc[:,:,c:c+1,:] -> [1,10,1,1]
        wc = b.node("Slice", [partner_kc,
                             b.init(f"cas{c}a", np.array([c], np.int64)),
                             b.init(f"cas{c}b", np.array([c + 1], np.int64)),
                             b.init(f"cas{c}c", np.array([2], np.int64))], f"wc{c}")  # [1,10,1,1]
        Xc = b.node("Slice", ["input",
                             b.init(f"cax{c}a", np.array([c], np.int64)),
                             b.init(f"cax{c}b", np.array([c + 1], np.int64)),
                             b.init(f"cax{c}c", np.array([1], np.int64))], f"caXc{c}")  # [1,1,30,30]
        term = b.node("Mul", [wc, Xc], f"caterm{c}")  # [1,10,30,30]
        cen_anchor = term if cen_anchor is None else b.node("Add", [cen_anchor, term], "cen_anchor")
    # clip to {0,1}
    cen_anchor = b.node("Min", [cen_anchor, b.init("onef2", np.array(1.0, np.float32))], "cen_anchor_clip")

    # anchor[k] = max(arm_anchor, cen_anchor)  [1,10,30,30]
    anchor = b.node("Max", [arm_anchor, cen_anchor], "anchor")  # [1,10,30,30] f32 {0,1}

    # --- paint arms: dilate anchor[k] by k's shape kernel ---
    # arm_paint[k] = sum_s arm_shape_valid[s] * conv(anchor[k], ARMK[s])  (grouped)
    arm_paint_terms = []
    for s in range(4):
        w = np.broadcast_to(armkern(s).reshape(1, 1, 3, 3), (10, 1, 3, 3)).copy()
        ws = b.init(f"paintw{s}", w)
        conv = b.node("Conv", [anchor, ws], f"paintconv{s}", group=10, pads=[1, 1, 1, 1], kernel_shape=[3, 3])
        gated = b.node("Mul", [conv, arm_shape_valid[s]], f"paintg{s}")
        arm_paint_terms.append(gated)
    arm_paint = arm_paint_terms[0]
    for t in arm_paint_terms[1:]:
        arm_paint = b.node("Max", [arm_paint, t], "arm_paint")  # [1,10,30,30]
    arm_paint = b.node("Cast", [b.node("Greater", [arm_paint, b.init("half1", np.array(0.5, np.float32))], "arm_paint_g")], "arm_paint_b", to=TensorProto.FLOAT)

    # --- paint centers: center color cc at anchor[k] positions ---
    # cen_paint[c] = max over k of partner_kc[k,c]*anchor[k]. Build per channel c.
    # anchor [1,10,30,30] (over k). For channel c, weight = partner_kc[:,:,c,:] [1,10,1,1].
    cen_paint_cols = []
    for c in range(10):
        wc = b.node("Slice", [partner_kc,
                             b.init(f"cps{c}a", np.array([c], np.int64)),
                             b.init(f"cps{c}b", np.array([c + 1], np.int64)),
                             b.init(f"cps{c}c", np.array([2], np.int64))], f"cpwc{c}")  # [1,10,1,1]
        prod = b.node("Mul", [anchor, wc], f"cpprod{c}")  # [1,10,30,30]
        red = b.node("ReduceMax", [prod, b.init(f"cpax{c}", np.array([1], np.int64))], f"cpred{c}", keepdims=1)  # [1,1,30,30]
        cen_paint_cols.append(red)
    cen_paint = b.node("Concat", cen_paint_cols, "cen_paint", axis=1)  # [1,10,30,30]

    # --- compose output one-hot ---
    # painted (colors c>=1) = max(input, arm_paint, cen_paint), then channel 0
    # cleared wherever a color was painted (keeps in-grid background channel 0).
    out1 = b.node("Max", ["input", arm_paint], "out1")
    paintedoh = b.node("Max", [out1, cen_paint], "paintedoh")  # [1,10,30,30]

    # color presence at each cell = max over channels 1..9 of paintedoh
    color_only = b.node("Mul", [paintedoh, b.init("nonzero_ch2", nzch)], "color_only")  # zero ch0
    color_any = b.node("ReduceMax", [color_only,
                                     b.init("ax1c", np.array([1], np.int64))], "color_any", keepdims=1)  # [1,1,30,30]
    # new channel 0 = input ch0 AND NOT color_any
    notcolor = b.node("Sub", [one, color_any], "notcolor")  # 1 - color_any
    new_ch0 = b.node("Mul", [ch0, notcolor], "new_ch0")  # [1,1,30,30]
    # assemble: ch0 replaced. Slice channels 1..9 from color_only, concat new_ch0.
    ch1_9 = b.node("Slice", [paintedoh,
                            b.init("c19a", np.array([1], np.int64)),
                            b.init("c19b", np.array([10], np.int64)),
                            b.init("c19c", np.array([1], np.int64))], "ch1_9")  # [1,9,30,30]
    outoh = b.node("Concat", [new_ch0, ch1_9], "outoh", axis=1)  # [1,10,30,30]
    out = b.node("Greater", [outoh, b.init("halff", np.array(0.5, np.float32))], "out_g")

    graph = helper.make_graph(
        b.nodes, "task173",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
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
