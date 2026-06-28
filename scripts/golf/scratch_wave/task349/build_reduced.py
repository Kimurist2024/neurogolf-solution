#!/usr/bin/env python3
"""Golf the incumbent task349 net by dropping the radius-1 detector channel.

Reuses the incumbent's proven QLinearConv weights/pads, but removes the radius-1
output channel from both convs (hpos: 5->4 ch; haloW: 5->4 ch) and instead obtains
the radius-1 halo for free by dilating maroon by 1 (MaxPool 3x3). A radius-1 (2x2)
square dilates to its exact 4x4 halo; larger squares only gain their inner ring,
which is a subset of their (already painted) halo -> harmless. Shrinks the 5-channel
hpos intermediate (4350 bytes) to 4 channels (~3480 bytes).
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
INC = HERE / "incumbent.onnx"
OUT = HERE / "cand.onnx"
H = W = 30
U8 = TensorProto.UINT8


def main():
    m = onnx.load(str(INC))
    d = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}

    # Drop radius-1 (channel index 0) from detector + halo convs.
    hW = d["hW"][1:].copy()      # [4,1,1,12]
    hB = d["hB"][1:].copy()      # [4]
    haloW = d["haloW"][:, 1:].copy()  # [1,4,11,20]

    b_nodes = []
    b_inits = []
    n = [0]

    def init(name, arr):
        b_inits.append(numpy_helper.from_array(arr, name)); return name

    def nd(op, ins, p, **attrs):
        n[0] += 1; out = f"{p}_{n[0]}"
        b_nodes.append(helper.make_node(op, ins, [out], name=out, **attrs)); return out

    idx9 = init("idx9", d["idx9"])
    idx0 = init("idx0", d["idx0"])
    x_scale = init("x_scale", d["x_scale"])
    x_zp = init("x_zp", d["x_zp"])
    w_zp = init("w_zp_i8", d["w_zp_i8"])
    one_u8 = init("one_u8", d["one_u8"])
    hW_i = init("hW", hW)
    hB_i = init("hB", hB)
    haloW_i = init("haloW", haloW)
    haloB_i = init("haloB", d["haloB"])
    zero_ch = init("zero_ch", d["zero_ch"])

    ch9f = nd("Gather", ["input", idx9], "ch9f", axis=1)
    bgf = nd("Gather", ["input", idx0], "bgf", axis=1)
    ch9 = nd("Cast", [ch9f], "ch9u", to=U8)
    bg = nd("Cast", [bgf], "bgu", to=U8)

    hpos = nd("QLinearConv",
              [ch9, x_scale, x_zp, hW_i, x_scale, w_zp, x_scale, x_zp, hB_i],
              "hpos", pads=[0, 1, 0, 9])
    halou = nd("QLinearConv",
               [hpos, x_scale, x_zp, haloW_i, x_scale, x_zp, x_scale, x_zp, haloB_i],
               "halou", pads=[5, 14, 5, 6])
    hmask = nd("Min", [halou, one_u8], "hmask")
    # radius-1 halo via maroon dilation by 1
    dil1 = nd("MaxPool", [ch9], "dil1", kernel_shape=[3, 3], pads=[1, 1, 1, 1],
              strides=[1, 1])
    green = nd("Max", [hmask, dil1], "green")

    beam = nd("MaxPool", [ch9], "beamu", kernel_shape=[30, 1], pads=[29, 0, 0, 0],
              strides=[1, 1])

    ch3 = nd("Mul", [green, bg], "ch3u")
    bgnh = nd("Sub", [bg, ch3], "bgnh")
    ch1 = nd("Mul", [beam, bgnh], "ch1u")
    ch0 = nd("Sub", [bgnh, ch1], "ch0u")

    parts = [ch0, ch1, zero_ch, ch3, zero_ch, zero_ch, zero_ch, zero_ch, zero_ch, ch9]
    b_nodes.append(helper.make_node("Concat", parts, ["output"], name="out", axis=1))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, H, W])
    outp = helper.make_tensor_value_info("output", U8, [1, 10, H, W])
    g = helper.make_graph(b_nodes, "task349_reduced", [inp], [outp], b_inits)
    mm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mm.ir_version = 8
    onnx.checker.check_model(mm, full_check=True)
    onnx.save(mm, str(OUT))
    print("saved", OUT)


if __name__ == "__main__":
    main()
