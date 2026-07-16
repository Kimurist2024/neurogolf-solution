#!/usr/bin/env python3
"""Build exact/generator-support-exact residual task349 variants."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others/71407/task349.onnx"
EXPECTED_SOURCE_SHA = "f7531b66a5399973ed57835584023c5bf1f61966c218b283cb721ba7ca45c8e2"
CANDIDATES = HERE / "candidates"

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path) -> dict[str, int]:
    memory, params, total = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(total)}


def check_and_save(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    onnx.save(model, path)


def pow_tables(source: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    removed = {"shift_by_mod", "valid_cols_table", "five_i32", "axis2_scalar"}
    kept = [x for x in model.graph.initializer if x.name not in removed]
    if len(kept) + len(removed) != len(model.graph.initializer):
        raise AssertionError("pow-table initializer set mismatch")
    kept.append(numpy_helper.from_array(np.asarray(2, dtype=np.int32), "two_i32"))
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    nodes = []
    removed_side_factor = replaced_valid = replaced_shift = 0
    for old in model.graph.node:
        node = copy.deepcopy(old)
        output = node.output[0] if node.output else ""
        if output == "side_factor":
            removed_side_factor += 1
            continue
        if output == "valid_cols":
            if node.op_type != "Gather" or list(node.input) != ["valid_cols_table", "side_factor"]:
                raise AssertionError("unexpected valid_cols producer")
            nodes.extend([
                helper.make_node(
                    "Pow", ["two_i32", "side"], ["valid_cols_plus1"],
                    name="valid_cols_plus1_from_side",
                ),
                helper.make_node(
                    "Sub", ["valid_cols_plus1", "one_i32"], ["valid_cols"],
                    name="valid_cols_from_side",
                ),
            ])
            replaced_valid += 1
            continue
        if output == "shift_factor":
            if node.op_type != "Gather" or list(node.input) != ["shift_by_mod", "radius_code"]:
                raise AssertionError("unexpected shift producer")
            nodes.append(helper.make_node(
                "Pow", ["two_i32", "hend_offset_i8"], ["shift_factor"],
                name="shift_factor_from_radius",
            ))
            replaced_shift += 1
            continue
        if node.op_type == "CumSum" and list(node.input) == [node.input[0], "axis2_scalar"]:
            node.input[1] = "two_i32"
        nodes.append(node)
    if (removed_side_factor, replaced_valid, replaced_shift) != (1, 1, 1):
        raise AssertionError((removed_side_factor, replaced_valid, replaced_shift))
    if any("axis2_scalar" in node.input for node in nodes):
        raise AssertionError("axis2_scalar remains live")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def direct_i8_side(source: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    arrays = {x.name: np.asarray(numpy_helper.to_array(x)) for x in model.graph.initializer}
    coords = arrays["coords4"]
    if coords.dtype != np.int32 or coords.min() != 0 or coords.max() != 29:
        raise AssertionError("unexpected coords4")
    rewritten = []
    for initializer in model.graph.initializer:
        if initializer.name == "coords4":
            rewritten.append(numpy_helper.from_array(coords.astype(np.int8), "coords4"))
        else:
            rewritten.append(copy.deepcopy(initializer))
    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)

    nodes = []
    replaced_side = removed_i8 = 0
    for old in model.graph.node:
        node = copy.deepcopy(old)
        output = node.output[0] if node.output else ""
        if output == "side":
            if node.op_type != "Cast" or list(node.input) != ["side_f"]:
                raise AssertionError("unexpected side producer")
            node.output[0] = "side_i8"
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = TensorProto.INT8
            replaced_side += 1
            nodes.append(node)
            continue
        if output == "side_i8":
            if node.op_type != "Cast" or list(node.input) != ["side"]:
                raise AssertionError("unexpected side_i8 producer")
            removed_i8 += 1
            continue
        for index, value in enumerate(node.input):
            if value == "side":
                node.input[index] = "side_i8"
        nodes.append(node)
    if (replaced_side, removed_i8) != (1, 1):
        raise AssertionError((replaced_side, removed_i8))
    if any("side" in node.input or "side" in node.output for node in nodes):
        raise AssertionError("legacy side tensor remains")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def direct_i8_topk(source: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    nodes = []
    removed_cast = rewired_topk = 0
    for old in model.graph.node:
        node = copy.deepcopy(old)
        output = node.output[0] if node.output else ""
        if output == "bottom_true_f16":
            if node.op_type != "Cast" or list(node.input) != ["bottom_true"]:
                raise AssertionError("unexpected bottom_true Cast")
            removed_cast += 1
            continue
        if node.name == "beam_bottoms":
            if node.op_type != "TopK" or node.input[0] != "bottom_true_f16":
                raise AssertionError("unexpected beam TopK")
            node.input[0] = "bottom_true"
            node.output[0] = "beam_bottoms_i8"
            rewired_topk += 1
        nodes.append(node)
    if (removed_cast, rewired_topk) != (1, 1):
        raise AssertionError((removed_cast, rewired_topk))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def valid_cols_from_affine_width(source: onnx.ModelProto) -> onnx.ModelProto:
    """Use width_factor[-side] == -2**side, then bitwise-not it."""
    model = copy.deepcopy(source)
    removed = {"valid_cols_table", "five_i32"}
    kept = [x for x in model.graph.initializer if x.name not in removed]
    if len(kept) + len(removed) != len(model.graph.initializer):
        raise AssertionError("valid-cols initializer set mismatch")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    nodes = []
    removed_factor = replaced_valid = 0
    for old in model.graph.node:
        node = copy.deepcopy(old)
        output = node.output[0] if node.output else ""
        if output == "side_factor":
            if node.op_type != "Div" or list(node.input) != ["side", "five_i32"]:
                raise AssertionError("unexpected side_factor producer")
            removed_factor += 1
            continue
        if output == "valid_cols":
            if node.op_type != "Gather" or list(node.input) != ["valid_cols_table", "side_factor"]:
                raise AssertionError("unexpected valid_cols producer")
            nodes.extend([
                helper.make_node("Neg", ["side"], ["neg_side_index"], name="neg_side_index"),
                helper.make_node(
                    "Gather", ["affine_width_factor", "neg_side_index"],
                    ["neg_valid_cols_plus1"], name="neg_valid_cols_plus1",
                ),
                helper.make_node(
                    "BitwiseNot", ["neg_valid_cols_plus1"], ["valid_cols"],
                    name="valid_cols_from_affine_width",
                ),
            ])
            replaced_valid += 1
            continue
        nodes.append(node)
    if (removed_factor, replaced_valid) != (1, 1):
        raise AssertionError((removed_factor, replaced_valid))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def shift_from_uint8(source: onnx.ModelProto) -> onnx.ModelProto:
    """Use the exhaustive radius table and shift one uint8 bit by radius."""
    model = copy.deepcopy(source)
    kept = [x for x in model.graph.initializer if x.name != "shift_by_mod"]
    if len(kept) + 1 != len(model.graph.initializer):
        raise AssertionError("shift table missing")
    kept.append(numpy_helper.from_array(np.asarray(1, dtype=np.uint8), "one_u8"))
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    nodes = []
    replaced = 0
    for old in model.graph.node:
        node = copy.deepcopy(old)
        output = node.output[0] if node.output else ""
        if output == "shift_factor":
            if node.op_type != "Gather" or list(node.input) != ["shift_by_mod", "radius_code"]:
                raise AssertionError("unexpected shift_factor producer")
            nodes.extend([
                helper.make_node(
                    "Cast", ["hend_offset_i8"], ["radius_u8"],
                    name="radius_u8_for_shift", to=TensorProto.UINT8,
                ),
                helper.make_node(
                    "BitShift", ["one_u8", "radius_u8"], ["shift_u8"],
                    name="shift_u8_from_radius", direction="LEFT",
                ),
                helper.make_node(
                    "Cast", ["shift_u8"], ["shift_factor"],
                    name="shift_factor_i32", to=TensorProto.INT32,
                ),
            ])
            replaced += 1
            continue
        nodes.append(node)
    if replaced != 1:
        raise AssertionError(replaced)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def rank4_zero_eliminate_beam_unsqueeze(source: onnx.ModelProto) -> onnx.ModelProto:
    """Use rank-4 max29 and positive-side Min to eliminate beam Unsqueeze."""
    model = copy.deepcopy(source)
    arrays = {x.name: np.asarray(numpy_helper.to_array(x)) for x in model.graph.initializer}
    max29 = arrays.get("max29_i8")
    if max29 is None or max29.dtype != np.int8 or max29.shape != () or int(max29) != 29:
        raise AssertionError("unexpected max29_i8")
    rewritten = []
    removed_unsq = 0
    for initializer in model.graph.initializer:
        if initializer.name == "unsq4":
            removed_unsq += 1
        else:
            rewritten.append(copy.deepcopy(initializer))
    if removed_unsq != 1:
        raise AssertionError("unsq4 missing")
    rewritten.append(numpy_helper.from_array(
        np.full((1, 1, 1, 1), 29, dtype=np.int8), "max29_rank4_i8"
    ))
    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)

    nodes = []
    replaced_clip = removed_node = rewired = 0
    for old in model.graph.node:
        node = copy.deepcopy(old)
        output = node.output[0] if node.output else ""
        if output == "beam_end_scalar_i8":
            if node.op_type != "Clip" or list(node.input) != ["side_i8", "zero_i8", "max29_i8"]:
                raise AssertionError("unexpected beam-end Clip")
            node = helper.make_node(
                "Min", ["side_i8", "max29_rank4_i8"], ["beam_end_scalar_i8"],
                name="beam_end_scalar_i8_from_positive_side",
            )
            replaced_clip += 1
        if output == "beam_end_index_i8":
            if node.op_type != "Unsqueeze" or list(node.input) != ["beam_end_scalar_i8", "unsq4"]:
                raise AssertionError("unexpected beam-end Unsqueeze")
            removed_node += 1
            continue
        for index, value in enumerate(node.input):
            if value == "beam_end_index_i8":
                node.input[index] = "beam_end_scalar_i8"
                rewired += 1
        nodes.append(node)
    if (replaced_clip, removed_node, rewired) != (1, 1, 1):
        raise AssertionError((replaced_clip, removed_node, rewired))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def narrow_side_and_negative_index(source: onnx.ModelProto) -> onnx.ModelProto:
    """Keep side in int8; widen only its negative Gather index."""
    model = copy.deepcopy(source)
    arrays = {x.name: np.asarray(numpy_helper.to_array(x)) for x in model.graph.initializer}
    coords = arrays.get("coords4")
    if coords is None or coords.dtype != np.int32 or coords.min() != 0 or coords.max() != 29:
        raise AssertionError("unexpected coords4")
    rewritten = []
    for initializer in model.graph.initializer:
        if initializer.name == "coords4":
            rewritten.append(numpy_helper.from_array(coords.astype(np.int8), "coords4"))
        else:
            rewritten.append(copy.deepcopy(initializer))
    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)

    nodes = []
    replaced_side = removed_side_i8 = replaced_neg = rewired_rows = 0
    for old in model.graph.node:
        node = copy.deepcopy(old)
        output = node.output[0] if node.output else ""
        if output == "side":
            if node.op_type != "Cast" or list(node.input) != ["side_f"]:
                raise AssertionError("unexpected side producer")
            node.output[0] = "side_i8"
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = TensorProto.INT8
            replaced_side += 1
            nodes.append(node)
            continue
        if output == "side_i8":
            if node.op_type != "Cast" or list(node.input) != ["side"]:
                raise AssertionError("unexpected side_i8 producer")
            removed_side_i8 += 1
            continue
        if output == "neg_side_index":
            if node.op_type != "Neg" or list(node.input) != ["side"]:
                raise AssertionError("unexpected negative-side producer")
            nodes.extend([
                helper.make_node("Neg", ["side_i8"], ["neg_side_index_i8"], name="neg_side_index_i8"),
                helper.make_node(
                    "Cast", ["neg_side_index_i8"], ["neg_side_index"],
                    name="neg_side_index_i32", to=TensorProto.INT32,
                ),
            ])
            replaced_neg += 1
            continue
        if node.name == "valid_rows4":
            if node.op_type != "Less" or list(node.input) != ["coords4", "side"]:
                raise AssertionError("unexpected row-validity producer")
            node.input[1] = "side_i8"
            rewired_rows += 1
        if "side" in node.input or "side" in node.output:
            raise AssertionError(f"unhandled legacy side use: {node.name}")
        nodes.append(node)
    if (replaced_side, removed_side_i8, replaced_neg, rewired_rows) != (1, 1, 1, 1):
        raise AssertionError((replaced_side, removed_side_i8, replaced_neg, rewired_rows))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def split_special_h_patch(source: onnx.ModelProto) -> onnx.ModelProto:
    """Split the duplicated special signature into one shared scalar condition."""
    model = copy.deepcopy(source)
    arrays = {x.name: np.asarray(numpy_helper.to_array(x)) for x in model.graph.initializer}
    indices = arrays.get("h_patch_indices_i8")
    sigs = arrays.get("h_patch_sigs")
    values = arrays.get("h_patch_values")
    if indices is None or sigs is None or values is None:
        raise AssertionError("h-patch initializers missing")
    flat_i = indices.reshape(-1)
    flat_s = sigs.reshape(-1)
    flat_v = values.reshape(-1)
    if not (
        flat_i.tolist() == [9, 12, 5, 8, 19, 27]
        and flat_s.tolist() == [495564, 495564, 133111344, 133111344, 214431744, 214431744]
        and flat_v.tolist() == [-24576, 24576, -16384, 16384, 63, -63]
    ):
        raise AssertionError("unexpected h-patch tables")
    replacements = {
        "h_patch_indices_i8": numpy_helper.from_array(indices[:, :, :4, :].copy(), "h_patch_indices_i8"),
        "h_patch_sigs": numpy_helper.from_array(sigs[:, :, :4, :].copy(), "h_patch_sigs"),
        "h_patch_values": numpy_helper.from_array(values[:, :, :4, :].copy(), "h_patch_values"),
    }
    rewritten = [replacements.get(x.name, copy.deepcopy(x)) for x in model.graph.initializer]
    rewritten.extend([
        numpy_helper.from_array(np.asarray(214431744, dtype=np.int32), "special_patch_sig"),
        numpy_helper.from_array(indices[:, :, 4:, :].copy(), "special_h_indices_i8"),
        numpy_helper.from_array(values[:, :, 4:, :].copy(), "special_h_values"),
    ])
    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)

    nodes = []
    rewired_indices = rewired_updates = replaced_condition = rewired_beam = 0
    for old in model.graph.node:
        node = copy.deepcopy(old)
        output = node.output[0] if node.output else ""
        if output == "halo_indices_i8":
            if node.op_type != "Concat" or list(node.input) != ["halo_start30", "halo_end30", "h_patch_indices_i8"]:
                raise AssertionError("unexpected halo index concat")
            node.input.append("special_h_indices_i8")
            rewired_indices += 1
        if output == "patch_sumR":
            nodes.append(node)
            nodes.extend([
                helper.make_node(
                    "Equal", ["patch_sumR", "special_patch_sig"], ["special_patch_cond"],
                    name="special_patch_cond",
                ),
                helper.make_node(
                    "Where", ["special_patch_cond", "special_h_values", "zero_i32"],
                    ["special_h_updates"], name="special_h_updates",
                ),
            ])
            continue
        if output == "halo_updates":
            if node.op_type != "Concat" or list(node.input) != ["X", "neg_X_stop", "h_patch_updates"]:
                raise AssertionError("unexpected halo update concat")
            node.input.append("special_h_updates")
            rewired_updates += 1
        if output == "sp_has_sig":
            if node.op_type != "Gather":
                raise AssertionError("unexpected special condition producer")
            replaced_condition += 1
            continue
        for index, value in enumerate(node.input):
            if value == "sp_has_sig":
                node.input[index] = "special_patch_cond"
                rewired_beam += 1
        nodes.append(node)
    if (rewired_indices, rewired_updates, replaced_condition, rewired_beam) != (1, 1, 1, 1):
        raise AssertionError((rewired_indices, rewired_updates, replaced_condition, rewired_beam))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def main() -> None:
    if digest(SOURCE) != EXPECTED_SOURCE_SHA:
        raise RuntimeError(f"source SHA changed: {digest(SOURCE)}")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    authority = onnx.load(SOURCE)
    variants: list[tuple[str, onnx.ModelProto]] = []
    valid_only = valid_cols_from_affine_width(authority)
    variants.append(("task349_valid_affine_only.onnx", valid_only))
    shift_only = shift_from_uint8(authority)
    variants.append(("task349_shift_u8_only.onnx", shift_only))
    combined = shift_from_uint8(valid_cols_from_affine_width(authority))
    variants.append(("task349_residual_combined.onnx", combined))
    broadcast_only = rank4_zero_eliminate_beam_unsqueeze(authority)
    variants.append(("task349_rank4_zero_only.onnx", broadcast_only))
    final = rank4_zero_eliminate_beam_unsqueeze(combined)
    variants.append(("task349_residual_final.onnx", final))
    narrow = narrow_side_and_negative_index(combined)
    variants.append(("task349_residual_narrow_side.onnx", narrow))
    narrow_final = rank4_zero_eliminate_beam_unsqueeze(narrow)
    variants.append(("task349_residual_narrow_final.onnx", narrow_final))
    patch_final = split_special_h_patch(narrow_final)
    variants.append(("task349_residual_patch_final.onnx", patch_final))

    rows = []
    for name, model in variants:
        path = CANDIDATES / name
        check_and_save(model, path)
        rows.append({"name": name, "sha256": digest(path), "profile": profile(path)})
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE),
        "source_profile": profile(SOURCE),
        "variants": rows,
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
