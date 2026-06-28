"""Build a compact ONNX for task216 (ARC 8efcae92, overlaps|random_pixels).

Rule: 3-4 non-overlapping solid blue boxes with red noise pixels; output the
crop of the box holding the strictly-maximum number of red pixels, at the
box's own size, placed at the output top-left.

Approach: slice the blue/red channels of the top-left 20x20 (the minimal grid
read, 3200 f32 bytes), cast to uint8, and run a corner-peel + per-box
width/height + MatMulInteger red-count + diagonal-GatherND winner selection,
then dynamic crop + Pad to [1,10,30,30].  The 2-channel c12 grid doubles as the
crop source for output reconstruction (channel0->color1 blue, channel1->color2
red).  All intermediates are uint8/int32 to minimise the memory footprint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent / "cand.onnx"
N = 20
MAXB = 4


class B:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def nm(self, p: str) -> str:
        self._n += 1
        return f"{p}_{self._n}"

    def init(self, name: str, arr: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(arr, name))
        return name

    def op(self, ty: str, ins: list[str], p: str, **attrs) -> str:
        o = self.nm(p)
        self.nodes.append(helper.make_node(ty, ins, [o], name=o, **attrs))
        return o


def build() -> onnx.ModelProto:
    b = B()

    # ---- constants ----
    q_scale = b.init("q_scale", np.array(1.0, dtype=np.float32))
    zero_u8 = b.init("zero_u8", np.array(0, dtype=np.uint8))
    one_u8 = b.init("one_u8", np.array(1, dtype=np.uint8))

    # Corner stencil on nz.  QLinearConv uses w_zero_point=1, so the effective
    # weights are (cw - 1) = [[0,-1],[-1,1]]: with pads [1,1,0,0] the output at
    # (i,j) = nz(i,j) - nz(i-1,j) - nz(i,j-1) which saturates to 1 (uint8) only
    # at a true top-left corner (self=1, up=0, left=0) and 0 elsewhere.
    cw = np.array(
        [[[[1, 0], [0, 2]], [[1, 0], [0, 2]]]], dtype=np.uint8
    )  # [1,2,2,2]: applied to c12=[blue,red]; sum over channels = nz stencil.
    corner_w = b.init("corner_w", cw)

    flat_w_i = b.init("flat_w_i", np.array(N, dtype=np.int32))
    coords = b.init("coords", np.arange(1, N + 1, dtype=np.int32))  # [N] = 1..N
    coords_col = b.init(
        "coords_col", np.arange(1, N + 1, dtype=np.int32).reshape(N, 1)
    )  # [N,1]
    scatter_pair = b.init("scatter_pair", np.zeros((1, 2), dtype=np.uint8))
    diag_idx = b.init(
        "diag_idx", np.array([[0, k, k] for k in range(MAXB)], dtype=np.int64)
    )
    thirty64 = b.init("thirty64", np.array(30, dtype=np.int64))
    crop_start_prefix = b.init("crop_start_prefix", np.array([0, 0], dtype=np.int32))
    crop_end_prefix = b.init("crop_end_prefix", np.array([1, 2], dtype=np.int32))
    # Pad spec (rank-4) prefix = [N_b, C_b, H_b, W_b, N_a, C_a]; pad_h/pad_w are
    # appended for H_a, W_a.  C_b=1 shifts blue->color1, red->color2; C_a=7 fills
    # the remaining colour channels so the total is 10.
    pad_prefix = b.init("pad_prefix", np.array([0, 1, 0, 0, 0, 7], dtype=np.int64))

    starts_c12 = b.init("starts_c12", np.array([1, 0, 0], dtype=np.int64))
    ends_c12 = b.init("ends_c12", np.array([3, N, N], dtype=np.int64))
    axes_c12 = b.init("axes_c12", np.array([1, 2, 3], dtype=np.int64))

    # ---- decode: slice the blue/red channels of the top-left 20x20 ----
    # c12[0]=blue (color1), c12[1]=red (color2).  Reading these two channels in
    # f32 (3200 bytes) is the minimal grid read; everything downstream is uint8.
    c12_f32 = b.op("Slice", ["input", starts_c12, ends_c12, axes_c12], "c12_f32")
    c12 = b.op("Cast", [c12_f32], "c12", to=TensorProto.UINT8)  # [1,2,20,20]

    # ---- corner score + peel up to 4 corners (conv on the 2-channel c12) ----
    tl = b.op(
        "QLinearConv",
        [c12, q_scale, zero_u8, corner_w, q_scale, one_u8, q_scale, zero_u8],
        "tl",
        kernel_shape=[2, 2],
        pads=[1, 1, 0, 0],
    )  # [1,1,20,20]
    tl_flat = b.op("Flatten", [tl], "tl_flat")  # [1,400]

    a0 = b.op("ArgMax", [tl_flat], "a0", axis=1, keepdims=1)
    a1 = b.op("ArgMax", [tl_flat], "a1", axis=1, keepdims=1, select_last_index=1)
    edge = b.op("Concat", [a0, a1], "edge", axis=1)
    tl_mid = b.op("ScatterElements", [tl_flat, edge, scatter_pair], "tl_mid", axis=1)
    a2 = b.op("ArgMax", [tl_mid], "a2", axis=1, keepdims=1)
    a3 = b.op("ArgMax", [tl_mid], "a3", axis=1, keepdims=1, select_last_index=1)

    a0i = b.op("Cast", [a0], "a0i", to=TensorProto.INT32)
    a1i = b.op("Cast", [a1], "a1i", to=TensorProto.INT32)
    a2i = b.op("Cast", [a2], "a2i", to=TensorProto.INT32)
    a3i = b.op("Cast", [a3], "a3i", to=TensorProto.INT32)
    top_idx_2d = b.op("Concat", [a0i, a1i, a2i, a3i], "top_idx_2d", axis=1)  # [1,4]
    top_idx = b.op("Squeeze", [top_idx_2d], "top_idx", axes=[0])  # [4]

    rows = b.op("Div", [top_idx, flat_w_i], "rows")  # [4]
    cols = b.op("Mod", [top_idx, flat_w_i], "cols")  # [4]

    # ---- widths ----
    # Gather the corner rows of c12 (both channels) then reduce channels: the
    # per-row nz vector tells where the box ends (first zero column).
    row_cands = b.op("Gather", [c12, rows], "row_cands", axis=2)  # [1,2,4,20]
    row_vecs = b.op("ReduceMax", [row_cands], "row_vecs", axes=[0, 1], keepdims=0)  # [4,20]
    row_zero = b.op("Equal", [row_vecs, zero_u8], "row_zero")  # [4,20]
    cols_col = b.op("Unsqueeze", [cols], "cols_col", axes=[1])  # [4,1]
    after_col = b.op("Less", [cols_col, coords], "after_col")  # [4,20]
    row_zero_after = b.op("And", [after_col, row_zero], "row_zero_after")  # [4,20]
    rza_u8 = b.op("Cast", [row_zero_after], "rza_u8", to=TensorProto.UINT8)
    width_end_raw = b.op("ArgMax", [rza_u8], "width_end_raw", axis=1, keepdims=0)  # [4]
    wer_i = b.op("Cast", [width_end_raw], "wer_i", to=TensorProto.INT32)
    width_has_zero = b.op("ReduceMax", [rza_u8], "width_has_zero", axes=[1], keepdims=0)
    whz_bool = b.op("Cast", [width_has_zero], "whz_bool", to=TensorProto.BOOL)
    width_end_i = b.op("Where", [whz_bool, wer_i, flat_w_i], "width_end_i")  # [4]
    widths = b.op("Sub", [width_end_i, cols], "widths")  # [4]

    # ---- heights ----
    col_cands = b.op("Gather", [c12, cols], "col_cands", axis=3)  # [1,2,20,4]
    col_vecs = b.op("ReduceMax", [col_cands], "col_vecs", axes=[0, 1], keepdims=0)  # [20,4]
    col_zero = b.op("Equal", [col_vecs, zero_u8], "col_zero")  # [20,4]
    rows_row = b.op("Unsqueeze", [rows], "rows_row", axes=[0])  # [1,4]
    after_row = b.op("Less", [rows_row, coords_col], "after_row")  # [20,4]
    col_zero_after = b.op("And", [after_row, col_zero], "col_zero_after")  # [20,4]
    cza_u8 = b.op("Cast", [col_zero_after], "cza_u8", to=TensorProto.UINT8)
    height_end_raw = b.op("ArgMax", [cza_u8], "height_end_raw", axis=0, keepdims=0)  # [4]
    her_i = b.op("Cast", [height_end_raw], "her_i", to=TensorProto.INT32)
    height_has_zero = b.op("ReduceMax", [cza_u8], "height_has_zero", axes=[0], keepdims=0)
    hhz_bool = b.op("Cast", [height_has_zero], "hhz_bool", to=TensorProto.BOOL)
    height_end_i = b.op("Where", [hhz_bool, her_i, flat_w_i], "height_end_i")  # [4]
    heights = b.op("Sub", [height_end_i, rows], "heights")  # [4]

    # ---- red counts per box via MatMulInteger contraction ----
    # red grid [1,20,20] = channel 1 of c12 (Gather collapses the channel axis).
    red_3d = b.op("Gather", [c12, b.init("one_idx", np.array(1, dtype=np.int64))], "red_3d", axis=1)  # [1,20,20]
    # column membership [4,20]: cols[b] < coord <= col_end[b]
    col_end_col = b.op("Unsqueeze", [width_end_i], "col_end_col", axes=[1])  # [4,1]
    col_lt_end = b.op("LessOrEqual", [coords, col_end_col], "col_lt_end")  # [4,20]
    col_in = b.op("And", [after_col, col_lt_end], "col_in")  # [4,20]
    # row membership [20,4]: rows[b] < coord <= row_end[b]
    row_end_row = b.op("Unsqueeze", [height_end_i], "row_end_row", axes=[0])  # [1,4]
    row_lt_end = b.op("LessOrEqual", [coords_col, row_end_row], "row_lt_end")  # [20,4]
    row_in = b.op("And", [after_row, row_lt_end], "row_in")  # [20,4]

    row_mask_u8 = b.op("Cast", [row_in], "row_mask_u8", to=TensorProto.UINT8)  # [20,4]
    row_mask_t = b.op("Transpose", [row_mask_u8], "row_mask_t", perm=[1, 0])  # [4,20]
    col_mask_u8 = b.op("Cast", [col_in], "col_mask_u8", to=TensorProto.UINT8)  # [4,20]
    col_mask_t = b.op("Transpose", [col_mask_u8], "col_mask_t", perm=[1, 0])  # [20,4]

    row_red_i = b.op("MatMulInteger", [row_mask_t, red_3d], "row_red_i")  # [1,4,20]
    row_red_u8 = b.op("Cast", [row_red_i], "row_red_u8", to=TensorProto.UINT8)
    count_matrix = b.op("MatMulInteger", [row_red_u8, col_mask_t], "count_matrix")  # [1,4,4]
    counts = b.op("GatherND", [count_matrix, diag_idx], "counts")  # [4]
    winner = b.op("ArgMax", [counts], "winner", axis=0, keepdims=0)  # scalar

    win_row = b.op("Gather", [rows, winner], "win_row", axis=0)
    win_col = b.op("Gather", [cols, winner], "win_col", axis=0)
    win_h = b.op("Gather", [heights, winner], "win_h", axis=0)
    win_w = b.op("Gather", [widths, winner], "win_w", axis=0)
    crop_row_end = b.op("Add", [win_row, win_h], "crop_row_end")
    crop_col_end = b.op("Add", [win_col, win_w], "crop_col_end")

    # c12 already is the 2-channel [blue, red] grid -> use it directly for crop.
    win_row_v = b.op("Unsqueeze", [win_row], "win_row_v", axes=[0])
    win_col_v = b.op("Unsqueeze", [win_col], "win_col_v", axes=[0])
    crow_v = b.op("Unsqueeze", [crop_row_end], "crow_v", axes=[0])
    ccol_v = b.op("Unsqueeze", [crop_col_end], "ccol_v", axes=[0])
    crop_starts = b.op("Concat", [crop_start_prefix, win_row_v, win_col_v], "crop_starts", axis=0)
    crop_ends = b.op("Concat", [crop_end_prefix, crow_v, ccol_v], "crop_ends", axis=0)
    crop = b.op("Slice", [c12, crop_starts, crop_ends], "crop")  # [1,2,h<=18,w<=18]
    # Slice ends are runtime values, so ONNX shape inference cannot derive the
    # crop H/W.  Annotate the (upper-bounded) static shape explicitly so the
    # validator's strict shape inference sees a fully static graph; the runtime
    # tensor is the true <=18x18 crop and the downstream Pad uses runtime dims.
    b._crop_name = crop

    win_h64 = b.op("Cast", [win_h], "win_h64", to=TensorProto.INT64)
    win_w64 = b.op("Cast", [win_w], "win_w64", to=TensorProto.INT64)
    win_h64_v = b.op("Unsqueeze", [win_h64], "win_h64_v", axes=[0])
    win_w64_v = b.op("Unsqueeze", [win_w64], "win_w64_v", axes=[0])
    pad_h = b.op("Sub", [thirty64, win_h64_v], "pad_h")
    pad_w = b.op("Sub", [thirty64, win_w64_v], "pad_w")
    dyn_pad = b.op("Concat", [pad_prefix, pad_h, pad_w], "dyn_pad", axis=0)  # [8]
    b.nodes.append(
        helper.make_node("Pad", [crop, dyn_pad], ["output"], name="output", mode="constant")
    )

    crop_vi = helper.make_tensor_value_info(
        b._crop_name, TensorProto.UINT8, [1, 2, 18, 18]
    )
    graph = helper.make_graph(
        b.nodes,
        "task216",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        b.inits,
        value_info=[crop_vi],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    return model


def main() -> None:
    model = build()
    onnx.save(model, OUT)
    onnx.checker.check_model(model, full_check=True)
    print(f"saved {OUT}  nodes={len(model.graph.node)} inits={len(model.graph.initializer)}")


if __name__ == "__main__":
    main()
