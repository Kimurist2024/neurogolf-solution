"""Rebuild task195 (80af3007): fractal 3x3 sprite self-product, margin-safe fp32.

out9[h][w] = a[h//3][w//3] * a[h%3][w%3]  (gray ch5); channel 0 = background in 9x9.

Keeps the efficient incumbent contraction (single sprite a, term axis via G), but
makes the channel-0 background EXACT and margin-clean by using a data-dependent
channel matrix C:
    term0 (a,a,G=I)     -> gray9                 (0/1)
    term1 (a,a,G=ones)  -> (sum a)^2 * valid9    (= s2 * valid9)
    C[5] = [1, 0]       -> gray9                 (active = 1.0)
    C[0] = [-s2, 1]     -> s2*valid - s2*gray    (= s2 on bg, exactly 0 on gray)
All active output cells are integers >= 1  ->  no cell in (0, 0.25), margin clean.
Slice is done on a uint8 cast of the input (cheap memory vs the fp32 incumbent slice).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts" / "cost500" / "agent3" / "task195.onnx"


def _np(name, arr, dt):
    return numpy_helper.from_array(np.asarray(arr, dtype=dt), name=name)


def build() -> onnx.ModelProto:
    nodes, inits = [], []

    # --- extract gray (ch5) sprite a [1,1,3,3] (uint8 slice) ---
    inits.append(_np("s_starts", [0, 5, 3, 3], np.int64))
    inits.append(_np("s_ends", [1, 6, 30, 30], np.int64))
    inits.append(_np("s_axes", [0, 1, 2, 3], np.int64))
    inits.append(_np("s_steps", [1, 1, 3, 3], np.int64))
    nodes.append(helper.make_node("Cast", ["input"], ["in_u8"], to=TensorProto.UINT8))
    nodes.append(helper.make_node("Slice", ["in_u8", "s_starts", "s_ends", "s_axes", "s_steps"], ["xg"]))
    nodes.append(helper.make_node("ReduceMax", ["xg"], ["row_has"], axes=[0, 1, 3], keepdims=0))
    nodes.append(helper.make_node("ReduceMax", ["xg"], ["col_has"], axes=[0, 1, 2], keepdims=0))
    nodes.append(helper.make_node("ArgMax", ["row_has"], ["top_i64"], axis=0, keepdims=0))
    nodes.append(helper.make_node("ArgMax", ["col_has"], ["left_i64"], axis=0, keepdims=0))
    inits.append(_np("offs012", [0, 1, 2], np.int64))
    nodes.append(helper.make_node("Add", ["top_i64", "offs012"], ["row_idx"]))
    nodes.append(helper.make_node("Add", ["left_i64", "offs012"], ["col_idx"]))
    nodes.append(helper.make_node("Gather", ["xg", "col_idx"], ["samp_c"], axis=3))
    nodes.append(helper.make_node("Gather", ["samp_c", "row_idx"], ["a_u8"], axis=2))  # [1,1,3,3]
    nodes.append(helper.make_node("Cast", ["a_u8"], ["a"], to=TensorProto.FLOAT))

    # --- s2 = (sum a)^2 ---  (ReduceSum opset13: axes is an input)
    inits.append(_np("red_axes", [0, 1, 2, 3], np.int64))
    nodes.append(helper.make_node("ReduceSum", ["a", "red_axes"], ["s_sum"], keepdims=0))
    nodes.append(helper.make_node("Mul", ["s_sum", "s_sum"], ["s2"]))  # scalar

    # --- C[10,2] = Cbase + Cmask * s2 ---
    Cbase = np.zeros((10, 2), np.float32)
    Cbase[5, 0] = 1.0
    Cbase[0, 1] = 1.0
    Cmask = np.zeros((10, 2), np.float32)
    Cmask[0, 0] = -1.0
    inits.append(_np("Cbase", Cbase, np.float32))
    inits.append(_np("Cmask", Cmask, np.float32))
    nodes.append(helper.make_node("Mul", ["Cmask", "s2"], ["Cscaled"]))  # broadcast scalar
    nodes.append(helper.make_node("Add", ["Cbase", "Cscaled"], ["C"]))   # [10,2]

    # --- projections + term combiner (fp32) ---
    Bblk = np.zeros((30, 3), np.float32)
    Bmod = np.zeros((30, 3), np.float32)
    for h in range(9):
        Bblk[h, h // 3] = 1.0
        Bmod[h, h % 3] = 1.0
    inits.append(_np("Bblk", Bblk, np.float32))
    inits.append(_np("Bmod", Bmod, np.float32))
    G = np.zeros((2, 3, 3), np.float32)
    G[0] = np.eye(3, dtype=np.float32)
    G[1] = np.ones((3, 3), np.float32)
    inits.append(_np("G", G, np.float32))

    # --- fused einsum (incumbent's efficient contraction) -> output ---
    nodes.append(
        helper.make_node(
            "Einsum",
            ["a", "a", "C", "Bblk", "G", "Bmod", "G", "Bblk", "G", "Bmod", "G"],
            ["output"],
            equation="bnrs,bnuv,ct,hg,tgr,hp,tpu,wm,tms,wq,tqv->bchw",
        )
    )

    graph = helper.make_graph(
        nodes, "task195",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> int:
    m = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(OUT))
    print("saved", OUT, OUT.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
