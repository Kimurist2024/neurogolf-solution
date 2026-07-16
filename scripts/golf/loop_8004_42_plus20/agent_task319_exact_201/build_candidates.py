#!/usr/bin/env python3
"""Build current-authority-only exact task319 regolf probes.

This script never touches root submission artifacts.  Every transformation is
applied to the immutable task319 member of submission_base_8009.46.zip.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = ROOT / "submission_base_8009.46.zip"
ARCHIVE_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
AUTHORITY_SHA256 = "29d5bfe25f86b18e0b5938d85e4f38cca72c34d8aad6390bff43579124d0e391"

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def node(model: onnx.ModelProto, output: str) -> onnx.NodeProto:
    return next(item for item in model.graph.node if output in item.output)


def initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    return next(item for item in model.graph.initializer if item.name == name)


def value(model: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return next(item for item in values if item.name == name)


def set_type_shape(
    model: onnx.ModelProto, name: str, elem_type: int | None = None,
    shape: list[int] | tuple[int, ...] | None = None,
) -> None:
    item = value(model, name)
    tensor = item.type.tensor_type
    if elem_type is not None:
        tensor.elem_type = int(elem_type)
    if shape is not None:
        del tensor.shape.dim[:]
        for size in shape:
            tensor.shape.dim.add().dim_value = int(size)


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    old = initializer(model, name)
    index = list(model.graph.initializer).index(old)
    del model.graph.initializer[index]
    model.graph.initializer.insert(index, numpy_helper.from_array(np.ascontiguousarray(array), name))


def add_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(array), name))


def set_attrs(item: onnx.NodeProto, **attrs: object) -> None:
    del item.attribute[:]
    for key, data in attrs.items():
        item.attribute.append(helper.make_attribute(key, data))


def add_vi(model: onnx.ModelProto, name: str, elem_type: int, shape: list[int]) -> None:
    model.graph.value_info.append(helper.make_tensor_value_info(name, elem_type, shape))


def dce(model: onnx.ModelProto) -> None:
    needed = {item.name for item in model.graph.output}
    keep: list[onnx.NodeProto] = []
    for item in reversed(model.graph.node):
        if any(name and name in needed for name in item.output):
            keep.append(item)
            needed.update(name for name in item.input if name)
    keep.reverse()
    del model.graph.node[:]
    model.graph.node.extend(keep)
    used = {name for item in model.graph.node for name in item.input if name}
    kept_inits = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)
    produced = {name for item in model.graph.node for name in item.output if name}
    kept_vi = [item for item in model.graph.value_info if item.name in produced]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)


def share_color_initializer(model: onnx.ModelProto) -> None:
    """Reuse safe_name_4 for the selected-color factor; remove colors10_u8.

    [1,10,1,1] and [3,1] broadcast to [1,10,3,1].  The two
    Einsums merely relabel the transposed color/selection axes.  The dynamic
    identity CenterCropPad retains the authority allocation lineage.
    """
    eq = node(model, "safe_name_35")
    del eq.input[:]
    eq.input.extend(["safe_name_4", "safe_name_34"])
    set_type_shape(model, "safe_name_35", TensorProto.BOOL, [1, 10, 3, 1])

    shape = node(model, "oh_cl_shape")
    set_attrs(shape, start=1, end=3)
    set_type_shape(model, "oh_cl_shape", TensorProto.INT64, [2])

    hidden = node(model, "oh_cl_hidden")
    set_attrs(hidden, axes=[1, 2])
    set_type_shape(model, "oh_cl_hidden", TensorProto.BOOL, [1, 1, 1, 1])
    set_type_shape(model, "oh_cl_f16", TensorProto.FLOAT16, [1, 1, 1, 1])
    set_type_shape(model, "oh_cl_f32", TensorProto.FLOAT, [1, 1, 1, 1])

    set_attrs(node(model, "abs_rows_f16"), equation="bchw,bckx,w->bkh")
    set_attrs(node(model, "row_score_f32"), equation="bchw,bckx,h->bk")


def transpose_correlation_drop_unsqueeze(model: onnx.ModelProto) -> None:
    """Transpose both 5x5 correlation operands and drop one Unsqueeze.

    Cross-correlation(A.T, B.T) is Cross-correlation(A, B).T.  The next node
    reduces both spatial axes, so the scalar maximum is invariant.
    """
    mask_view = node(model, "safe_name_78")
    del mask_view.input[:]
    mask_view.input.extend(["safe_name_2", "safe_name_5"])
    source_unpack = node(model, "safe_name_81")
    del source_unpack.input[:]
    source_unpack.input.extend(["safe_name_76", "safe_name_78"])

    replace_initializer(
        model, "safe_name_22",
        numpy_helper.to_array(initializer(model, "safe_name_22")).reshape(1, 1, 5, 1),
    )
    mag_unpack = node(model, "safe_name_86")
    mag_unpack.input[0] = "safe_name_84"


def minrow_direct_i32(model: onnx.ModelProto) -> None:
    """Cast the small row origin to int32 before Add and drop the recast."""
    cast = node(model, "minrow_flat_u8")
    cast.input[1] = "one_i64"  # Despite its historical name this initializer is int32.
    replace_initializer(
        model, "row_idx9_u8",
        numpy_helper.to_array(initializer(model, "row_idx9_u8")).astype(np.int32),
    )
    for name in ("minrow_flat_u8", "minrow_u8", "row_indices_u8"):
        set_type_shape(model, name, TensorProto.INT32, None)
    node(model, "gathered_abs_f16").input[1] = "row_indices_u8"


def swap_other_index_where(model: onnx.ModelProto) -> None:
    """Replace 3-index with the exact swapped boolean Where and drop [3]."""
    item = node(model, "other_idx")
    item.op_type = "Where"
    del item.input[:]
    item.input.extend(["cond1s", "two_i64", "one_i64"])
    set_attrs(item)
    set_type_shape(model, "other_idx", TensorProto.INT32, [])
    set_type_shape(model, "other_rows", TensorProto.UINT8, [1, 5])
    set_type_shape(model, "other_color", TensorProto.UINT8, [1])


def absorb_corr_qscale(model: onnx.ModelProto) -> None:
    """Use scale 4 and compare to C, instead of scale 8 and compare to 2*C.

    On generator support C is a non-background color count and is at most 100:
    a 5x5 source has at most 25 cells and its 2x magnification at most 100.
    The binary 5x5 overlap S is at most 25.  Thus 2*C and 8*S are both at
    most 200, so uint8 overflow/saturation cannot alter
    ``8*S >= 2*C iff 4*S >= C``.
    """
    replace_initializer(model, "eight_f32", np.asarray(4.0, dtype=np.float32))
    node(model, "safe_name_108").input[1] = "safe_name_100"


def remove_base_squeeze(model: onnx.ModelProto) -> None:
    """Keep the singleton condition rank and transpose the selected row vector."""
    node(model, "safe_name_109").input[0] = "safe_name_108"
    selected = node(model, "safe_name_113")
    selected.op_type = "Transpose"
    del selected.input[:]
    selected.input.extend(["safe_name_111"])
    set_attrs(selected, perm=[0, 1, 3, 2])
    set_type_shape(model, "safe_name_109", TensorProto.BOOL, [1, 1, 1, 1])
    set_type_shape(model, "safe_name_111", TensorProto.UINT8, [1, 1, 1, 5])
    set_type_shape(model, "safe_name_113", TensorProto.UINT8, [1, 1, 5, 1])
    set_type_shape(model, "safe_name_117", TensorProto.UINT8, [1, 1, 1, 1])


def terminal_scatter_weights(model: onnx.ModelProto) -> None:
    """Build the dynamic {-1,0,+1} 1x1 filters with two ScatterElements."""
    color0 = node(model, "color0")
    color0.input[0] = "safe_name_34_unsq"
    set_attrs(color0, axis=1)
    other = node(model, "other_color")
    other.input[0] = "safe_name_34_unsq"
    set_attrs(other, axis=1)
    set_type_shape(model, "color0", TensorProto.UINT8, [1, 1, 1, 1])
    other_index_shape = value(model, "other_idx").type.tensor_type.shape.dim
    if len(other_index_shape) == 0:
        set_type_shape(model, "other_color", TensorProto.UINT8, [1, 1, 1])
    else:
        set_type_shape(model, "other_color", TensorProto.UINT8, [1, 1, 1, 1])
    set_type_shape(model, "safe_name_117", TensorProto.UINT8, [1, 1, 1, 1])

    old = list(model.graph.node)
    insert_at = next(i for i, item in enumerate(old) if "eq_target2" in item.output)
    drop = {"eq_target2", "eq_bg2", "w_base2", "w_u8_2"}
    old = [item for item in old if not any(name in drop for name in item.output)]
    insert_at = next(i for i, item in enumerate(old) if "output" in item.output)

    target_cast = helper.make_node(
        "CastLike", ["safe_name_117", "one_i64"], ["target_idx_i32"], name="target_idx_i32"
    )
    scatter_bg = helper.make_node(
        "ScatterElements",
        ["weight_base_ones_u8", "safe_name_27", "weight_bg_zero_u8"],
        ["w_base2"], axis=0, name="w_base2",
    )
    scatter_target = helper.make_node(
        "ScatterElements",
        ["w_base2", "target_idx_i32", "weight_target_two_u8"],
        ["w_u8_2"], axis=0, name="w_u8_2",
    )
    old[insert_at:insert_at] = [target_cast, scatter_bg, scatter_target]
    del model.graph.node[:]
    model.graph.node.extend(old)

    add_initializer(model, "weight_base_ones_u8", np.ones((10, 1, 1, 1), dtype=np.uint8))
    add_initializer(model, "weight_bg_zero_u8", np.zeros((1, 1, 1, 1), dtype=np.uint8))
    add_initializer(model, "weight_target_two_u8", np.full((1, 1, 1, 1), 2, dtype=np.uint8))
    add_vi(model, "target_idx_i32", TensorProto.INT32, [1, 1, 1, 1])

    names_to_drop = {"eq_target2", "eq_bg2"}
    kept_vi = [item for item in model.graph.value_info if item.name not in names_to_drop]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)


def free_input_truthful_refactor(model: onnx.ModelProto) -> None:
    """Move the wide fp16 cast past the reducing operations.

    The scorer input is free.  Legal task319 grids occupy at most 19x19 in the
    top-left of the 30x30 tensor, so the last row/column are zero exactly as in
    the authority's 29x29 CenterCropPad.  Float32 reductions of one-hot values
    are exact; the small reduced tensors are then cast back to the authority's
    fp16 values.
    """
    drop_outputs = {
        "safe_name_23", "safe_name_24", "safe_name_25",
        "oh_cl_shape", "oh_cl_hidden", "oh_cl_f16",
    }
    kept = [item for item in model.graph.node if not any(out in drop_outputs for out in item.output)]

    counts = next(item for item in kept if "safe_name_26" in item.output)
    counts.input[0] = "input"
    # Keep the count/TopK path in float32.  Casting the 10-wide count tensor to
    # fp16 made ORT's CPU fp16 rewrite attempt an invalid buffer reuse between
    # the pre-TopK and broadcast TopK shapes.
    masked_counts = next(item for item in kept if "safe_name_30" in item.output)
    masked_counts.input[1] = "zero_f32"

    oh_f32 = next(item for item in kept if "oh_cl_f32" in item.output)
    oh_f32.input[0] = "safe_name_35"

    abs_rows = next(item for item in kept if "abs_rows_f16" in item.output)
    abs_rows.input[0] = "input"
    abs_rows.input[1] = "oh_cl_f32"
    abs_rows.output[0] = "abs_rows_f32"
    abs_cast = helper.make_node(
        "CastLike", ["abs_rows_f32", "safe_name_18"], ["abs_rows_f16"],
        name="abs_rows_to_f16",
    )
    row_score = next(item for item in kept if "row_score_f32" in item.output)
    row_score.input[0] = "input"

    # Put the selected-color cast before both Einsums.
    kept.remove(oh_f32)
    abs_index = kept.index(abs_rows)
    kept.insert(abs_index, oh_f32)
    kept.insert(kept.index(abs_rows) + 1, abs_cast)
    del model.graph.node[:]
    model.graph.node.extend(kept)

    abs_values = numpy_helper.to_array(initializer(model, "abs_pack_f16")).astype(np.float32)
    replace_initializer(model, "abs_pack_f16", np.concatenate([abs_values, np.zeros(1, np.float32)]))
    row_values = numpy_helper.to_array(initializer(model, "row_weight32_f32")).astype(np.float32)
    replace_initializer(model, "row_weight32_f32", np.concatenate([row_values, np.zeros(1, np.float32)]))
    corr_pattern = numpy_helper.to_array(initializer(model, "corr_pattern_f16")).astype(np.float32)
    replace_initializer(model, "corr_pattern_f16", corr_pattern)
    add_initializer(model, "zero_f32", np.asarray(0.0, dtype=np.float32))

    names_to_drop = drop_outputs | {"abs_rows_f16"}
    vi = [item for item in model.graph.value_info if item.name not in names_to_drop]
    del model.graph.value_info[:]
    model.graph.value_info.extend(vi)
    set_type_shape(model, "safe_name_26", TensorProto.FLOAT, [1, 10, 1, 1])
    set_type_shape(model, "safe_name_30", TensorProto.FLOAT, [1, 10, 1, 1])
    set_type_shape(model, "safe_name_32", TensorProto.FLOAT, [1, 3, 1, 1])
    set_type_shape(model, "safe_name_99", TensorProto.FLOAT, [1, 1, 1])
    add_vi(model, "abs_rows_f32", TensorProto.FLOAT, [1, 3, 30])
    add_vi(model, "abs_rows_f16", TensorProto.FLOAT16, [1, 3, 30])
    set_type_shape(model, "safe_name_35", TensorProto.BOOL, [3, 10])
    set_type_shape(model, "oh_cl_f32", TensorProto.FLOAT, [3, 10])
    set_type_shape(model, "row_score_f32", TensorProto.FLOAT, [1, 3])


TRANSFORMS = {
    "share_color_initializer": (share_color_initializer,),
    "transpose_corr": (transpose_correlation_drop_unsqueeze,),
    "minrow_i32": (minrow_direct_i32,),
    "other_idx_where": (swap_other_index_where,),
    "qscale4_no_shift": (absorb_corr_qscale,),
    "remove_base_squeeze": (remove_base_squeeze,),
    "terminal_scatter": (terminal_scatter_weights,),
    "combined_runnable": (
        transpose_correlation_drop_unsqueeze,
        swap_other_index_where,
        absorb_corr_qscale,
        remove_base_squeeze,
        terminal_scatter_weights,
    ),
    "combined": (
        share_color_initializer,
        transpose_correlation_drop_unsqueeze,
        minrow_direct_i32,
        swap_other_index_where,
        absorb_corr_qscale,
        remove_base_squeeze,
        terminal_scatter_weights,
    ),
}


def runtime_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    typed = {
        item.name: item
        for item in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for item in traced.graph.node:
        for output in item.output:
            if output and output in typed and output not in names:
                traced.graph.output.append(copy.deepcopy(typed[output]))
                names.append(output)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    example = scoring.load_examples(319)["train"][0]
    converted = scoring.convert_to_numpy(example)
    if converted is None:
        raise RuntimeError("task319 train[0] conversion failed")
    data = converted["input"]
    outputs = session.run(names, {session.get_inputs()[0].name: data})
    return {name: list(np.asarray(output).shape) for name, output in zip(names, outputs, strict=True)}


def honest_metadata(model: onnx.ModelProto) -> onnx.ModelProto:
    honest = copy.deepcopy(model)
    shapes = runtime_shapes(honest)
    for item in list(honest.graph.value_info) + list(honest.graph.output):
        if item.name in shapes:
            set_type_shape(honest, item.name, None, shapes[item.name])
    # Re-inference proves that repaired metadata and operator schemas agree.
    shape_inference.infer_shapes(honest, strict_mode=True, data_prop=True)
    return honest


def write(label: str, model: onnx.ModelProto, rows: list[dict[str, object]]) -> None:
    dce(model)
    data = model.SerializeToString()
    path = HERE / "candidates" / f"task319_{label}.onnx"
    path.write_bytes(data)
    rows.append({
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(data),
        "bytes": len(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params": sum(int(np.prod(item.dims)) if item.dims else 1 for item in model.graph.initializer),
    })


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "authority").mkdir(exist_ok=True)
    (HERE / "candidates").mkdir(exist_ok=True)
    if digest(ARCHIVE.read_bytes()) != ARCHIVE_SHA256:
        raise RuntimeError("immutable archive hash drift")
    with zipfile.ZipFile(ARCHIVE) as archive:
        payload = archive.read("task319.onnx")
    if digest(payload) != AUTHORITY_SHA256:
        raise RuntimeError("task319 authority hash drift")
    (HERE / "authority/task319.onnx").write_bytes(payload)
    authority = onnx.load_model_from_string(payload)

    rows: list[dict[str, object]] = []
    write("authority", copy.deepcopy(authority), rows)
    built: dict[str, onnx.ModelProto] = {}
    for label, transforms in TRANSFORMS.items():
        candidate = copy.deepcopy(authority)
        for transform in transforms:
            transform(candidate)
        write(label, candidate, rows)
        built[label] = candidate

    honest_errors: dict[str, str] = {}
    for source in ("combined_runnable", "combined"):
        try:
            honest = honest_metadata(built[source])
            write(f"{source}_honest_metadata", honest, rows)
        except Exception as exc:  # noqa: BLE001
            honest_errors[source] = f"{type(exc).__name__}: {exc}"

    try:
        free_input = copy.deepcopy(authority)
        free_input_truthful_refactor(free_input)
        dce(free_input)
        free_input = honest_metadata(free_input)
        write("truthful_free_input", free_input, rows)
    except Exception as exc:  # noqa: BLE001
        honest_errors["truthful_free_input"] = f"{type(exc).__name__}: {exc}"

    report = {
        "archive": "submission_base_8009.46.zip",
        "archive_sha256": ARCHIVE_SHA256,
        "authority_sha256": AUTHORITY_SHA256,
        "candidates": rows,
        "honest_metadata_errors": honest_errors,
    }
    (HERE / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
