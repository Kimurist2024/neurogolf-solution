#!/usr/bin/env python3
"""Build exact residual task319 probes from the staged cost-978 authority."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others/71407/task319.onnx"
SOURCE_SHA256 = "ade6b708b4ee6a0ba65d19e4182748750514435b3b8a005289582154b7208fd4"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def node(model: onnx.ModelProto, output: str) -> onnx.NodeProto:
    return next(item for item in model.graph.node if output in item.output)


def initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    return next(item for item in model.graph.initializer if item.name == name)


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    old = initializer(model, name)
    index = list(model.graph.initializer).index(old)
    del model.graph.initializer[index]
    model.graph.initializer.insert(index, numpy_helper.from_array(np.ascontiguousarray(array), name))


def set_attrs(item: onnx.NodeProto, **attrs: object) -> None:
    del item.attribute[:]
    for key, value in attrs.items():
        item.attribute.append(helper.make_attribute(key, value))


def dce(model: onnx.ModelProto) -> None:
    needed = {item.name for item in model.graph.output}
    kept: list[onnx.NodeProto] = []
    for item in reversed(model.graph.node):
        if any(output and output in needed for output in item.output):
            kept.append(item)
            needed.update(name for name in item.input if name)
    kept.reverse()
    del model.graph.node[:]
    model.graph.node.extend(kept)
    used = {name for item in model.graph.node for name in item.input if name}
    inits = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)
    produced = {name for item in model.graph.node for name in item.output if name}
    values = [item for item in model.graph.value_info if item.name in produced]
    del model.graph.value_info[:]
    model.graph.value_info.extend(values)


def scatter_zero_argmax(model: onnx.ModelProto) -> None:
    """Replace Cast/Equal/Where background masking by one ScatterElements.

    safe_name_27 is ArgMax over the ten count channels.  Scattering a float16
    zero into safe_name_26 at that index is exactly the existing Where mask.
    Reshaping the zero initializer is free in parameter count and affects no
    CastLike consumer because CastLike only reads its element type.
    """
    zero = numpy_helper.to_array(initializer(model, "safe_name_18"))
    replace_initializer(model, "safe_name_18", zero.reshape(1, 1, 1, 1))
    item = node(model, "safe_name_30")
    item.op_type = "ScatterElements"
    del item.input[:]
    item.input.extend(["safe_name_26", "safe_name_27", "safe_name_18"])
    set_attrs(item, axis=1)
    dce(model)


def direct_static_crop_shape(model: onnx.ModelProto) -> None:
    """Replace dynamic scalar-shape broadcasting by explicit [29,29].

    The first CenterCropPad yields the one-element tensor [29]; the second
    CenterCropPad broadcasts that target over axes [2,3].  Supplying [29,29]
    directly is the same crop and satisfies strict static inference.  It adds
    one parameter but removes an eight-byte node output.
    """
    replace_initializer(model, "shape23_const", np.asarray([29, 29], dtype=np.int64))
    crop = node(model, "safe_name_24")
    crop.input[1] = "shape23_const"
    dce(model)


def reduce_condition_direct_scalar(model: onnx.ModelProto) -> None:
    """Reduce the [1,1,2] equality tensor over all axes directly to scalar."""
    reduce = node(model, "cond1")
    reduce.output[0] = "cond1s"
    del reduce.input[1:]
    set_attrs(reduce, keepdims=0)
    # Remove the obsolete Squeeze producer before DCE sees duplicate outputs.
    kept = [item for item in model.graph.node if not (
        item.op_type == "Squeeze" and "cond1s" in item.output
    )]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    dce(model)


def terminal_background_where(model: onnx.ModelProto) -> None:
    """Reuse the existing background mask instead of a ten-one initializer.

    Transposing safe_name_29 gives the Conv-weight channel layout.  One Where
    then creates 0 at background and 1 elsewhere.  This costs one extra 10-byte
    mask but removes eleven parameters, for an exact one-point reduction.
    """
    old = list(model.graph.node)
    insert = next(i for i, item in enumerate(old) if "w_base2" in item.output)
    transpose = helper.make_node(
        "Transpose", ["safe_name_29"], ["bg_mask_w"],
        perm=[1, 0, 2, 3], name="bg_mask_w",
    )
    old.insert(insert, transpose)
    del model.graph.node[:]
    model.graph.node.extend(old)
    base = node(model, "w_base2")
    base.op_type = "Where"
    del base.input[:]
    base.input.extend(["bg_mask_w", "safe_name_13", "safe_name_14"])
    set_attrs(base)
    model.graph.value_info.append(
        helper.make_tensor_value_info("bg_mask_w", TensorProto.BOOL, [10, 1, 1, 1])
    )
    dce(model)


def equal_argmax_in_int64(model: onnx.ModelProto) -> None:
    """Compare ArgMax directly to an int64 color ramp and drop its uint8 Cast."""
    colors = numpy_helper.to_array(initializer(model, "safe_name_4")).astype(np.int64)
    replace_initializer(model, "safe_name_4", colors)
    compare = node(model, "safe_name_29")
    compare.input[1] = "safe_name_27"
    dce(model)


TRANSFORMS = {
    "scatter_zero_argmax": (scatter_zero_argmax,),
    "direct_static_crop_shape": (direct_static_crop_shape,),
    "reduce_condition_direct_scalar": (reduce_condition_direct_scalar,),
    "terminal_background_where": (terminal_background_where,),
    "equal_argmax_in_int64": (equal_argmax_in_int64,),
    "combined_cond_terminal": (
        reduce_condition_direct_scalar,
        terminal_background_where,
    ),
    "combined_best_local": (
        reduce_condition_direct_scalar,
        terminal_background_where,
        equal_argmax_in_int64,
    ),
    "combined_safe_local": (
        direct_static_crop_shape,
        reduce_condition_direct_scalar,
        terminal_background_where,
    ),
}


def main() -> None:
    payload = SOURCE.read_bytes()
    if digest(payload) != SOURCE_SHA256:
        raise RuntimeError("staged task319 authority hash drift")
    source = onnx.load_model_from_string(payload)
    rows = []
    for label, transforms in TRANSFORMS.items():
        candidate = copy.deepcopy(source)
        for transform in transforms:
            transform(candidate)
        data = candidate.SerializeToString()
        path = HERE / "candidates" / f"task319_{label}.onnx"
        path.write_bytes(data)
        rows.append({
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(data),
            "bytes": len(data),
            "nodes": len(candidate.graph.node),
            "initializers": len(candidate.graph.initializer),
            "params_declared": sum(int(np.prod(x.dims)) if x.dims else 1 for x in candidate.graph.initializer),
        })
    report = {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": SOURCE_SHA256, "candidates": rows}
    (HERE / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
