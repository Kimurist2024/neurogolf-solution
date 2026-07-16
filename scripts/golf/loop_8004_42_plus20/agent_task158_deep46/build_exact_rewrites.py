#!/usr/bin/env python3
"""Build only generator-independent, Boolean-exact task158 rewrites."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline/task158.onnx"
SOUND = HERE / "sound"


def remove_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def source_high_where() -> onnx.ModelProto:
    """Compute anchor_high & ~source_corner in one Where output.

    The baseline materializes source_high = source_corner & anchor_high and
    then XORs it out of anchor_high.  Because source_high is a subset of
    anchor_high, that is identically anchor_high where source_corner is false.
    """
    model = onnx.load(SOURCE)
    source_high_index = next(
        index
        for index, node in enumerate(model.graph.node)
        if list(node.output) == ["source_high_mask"]
    )
    partial_high_index = next(
        index
        for index, node in enumerate(model.graph.node)
        if list(node.output) == ["partial_high_mask"]
    )
    assert source_high_index < partial_high_index
    source_high = model.graph.node[source_high_index]
    partial_high = model.graph.node[partial_high_index]
    assert source_high.op_type == "And"
    assert list(source_high.input) == ["source_corner", "anchor_high"]
    assert partial_high.op_type == "Xor"
    assert list(partial_high.input) == ["anchor_high", "source_high_mask"]

    false_name = "exact_false"
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(False, dtype=np.bool_), false_name)
    )
    replacement = helper.make_node(
        "Where",
        ["source_corner", false_name, "anchor_high"],
        ["partial_high_mask"],
        name="partial_high_mask",
    )
    nodes = list(model.graph.node)
    nodes[source_high_index] = replacement
    del nodes[partial_high_index]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    remove_value_info(model, {"source_high_mask"})
    return model


def optional_topk_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Experimental ONNX optional-output probe for unused TopK values."""
    for node in model.graph.node:
        if node.output and node.output[0] in {"low_rank_values", "high_rank_values"}:
            node.output[0] = ""
    remove_value_info(model, {"low_rank_values", "high_rank_values"})
    return model


def scatter_max_invalid_zero() -> onnx.ModelProto:
    """Avoid five identity gathers; suppress invalid slots at ScatterElements.

    Valid low/high pairs are already ordered in slots 0..2, so gathering them
    with ``Where(p_valid, [0,1,2], 0)`` is the identity for every real object.
    The gathers only made unused slots duplicate slot zero.  Instead, keep the
    ordered tensors directly, set unused update codes to zero, and request the
    standard opset-18 ``max`` reduction so a zero duplicate cannot erase a
    real update.  This remains subject to runtime index-range and ORT-kernel
    verification before it can be considered SOUND.
    """
    model = onnx.load(SOURCE)
    replacements = {
        "br_obj_top": "box_top",
        "br_obj_left": "box_left",
        "obj_mag": "p_mag",
        "s2_obj_vb": "box_vflip",
        "s2_obj_hb": "box_hflip",
    }
    removed_outputs = {"s2_paint_order", *replacements.keys()}
    kept_nodes = [
        node
        for node in model.graph.node
        if not any(output in removed_outputs for output in node.output)
    ]
    for node in kept_nodes:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]

    code_index = next(
        index
        for index, node in enumerate(kept_nodes)
        if list(node.output) == ["pq_obj_code"]
    )
    gated_name = "pq_obj_code_valid"
    gated = helper.make_node(
        "Where",
        ["p_valid", "pq_obj_code", "pq_u8_zero"],
        [gated_name],
        name=gated_name,
    )
    kept_nodes.insert(code_index + 1, gated)
    for node in kept_nodes[code_index + 2 :]:
        for index, name in enumerate(node.input):
            if name == "pq_obj_code":
                node.input[index] = gated_name

    scatter = next(
        node for node in kept_nodes if list(node.output) == ["pq_seed_flat"]
    )
    assert scatter.op_type == "ScatterElements"
    scatter.attribute.extend([helper.make_attribute("reduction", "max")])

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    remove_value_info(model, removed_outputs)
    model.graph.value_info.append(
        helper.make_tensor_value_info(gated_name, onnx.TensorProto.UINT8, [1, 3])
    )
    return model


def scatter_max_orientation_only() -> onnx.ModelProto:
    """Remove only the two identity orientation gathers.

    Unlike :func:`scatter_max_invalid_zero`, this keeps the baseline's ordered
    top/left/magnitude tensors.  Consequently every unused slot still points
    inside the first real object's box and can never create an out-of-range
    ScatterElements index.  Its update value is gated to zero, and ``max``
    reduction makes such zero duplicates observationally inert.
    """
    model = onnx.load(SOURCE)
    replacements = {
        "s2_obj_vb": "box_vflip",
        "s2_obj_hb": "box_hflip",
    }
    removed_outputs = set(replacements)
    kept_nodes = [
        node
        for node in model.graph.node
        if not any(output in removed_outputs for output in node.output)
    ]
    for node in kept_nodes:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]

    code_index = next(
        index
        for index, node in enumerate(kept_nodes)
        if list(node.output) == ["pq_obj_code"]
    )
    gated_name = "pq_obj_code_valid"
    kept_nodes.insert(
        code_index + 1,
        helper.make_node(
            "Where",
            ["p_valid", "pq_obj_code", "pq_u8_zero"],
            [gated_name],
            name=gated_name,
        ),
    )
    for node in kept_nodes[code_index + 2 :]:
        for index, name in enumerate(node.input):
            if name == "pq_obj_code":
                node.input[index] = gated_name

    scatter = next(
        node for node in kept_nodes if list(node.output) == ["pq_seed_flat"]
    )
    assert scatter.op_type == "ScatterElements"
    scatter.attribute.extend([helper.make_attribute("reduction", "max")])

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    remove_value_info(model, removed_outputs)
    model.graph.value_info.append(
        helper.make_tensor_value_info(gated_name, onnx.TensorProto.UINT8, [1, 3])
    )
    return model


def drop_unused_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(value.name for value in model.graph.output)
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model


def validate_and_save(model: onnx.ModelProto, path: Path) -> dict[str, object]:
    row: dict[str, object] = {"path": str(path)}
    try:
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        nonstatic = [
            value.name
            for value in list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
            if any(
                not dim.HasField("dim_value") or dim.dim_value <= 0
                for dim in value.type.tensor_type.shape.dim
            )
        ]
        if nonstatic:
            raise ValueError(f"nonstatic inferred tensors: {nonstatic[:8]}")
        onnx.save(model, path)
        row.update(saved=True, checker=True, strict_data_prop=True)
    except Exception as exc:  # noqa: BLE001
        row.update(saved=False, error=f"{type(exc).__name__}: {exc}")
    return row


def main() -> None:
    SOUND.mkdir(parents=True, exist_ok=True)
    exact = source_high_where()
    rows = [
        validate_and_save(exact, SOUND / "task158_source_high_where.onnx"),
        validate_and_save(
            optional_topk_outputs(source_high_where()),
            SOUND / "task158_source_high_where_optional_topk.onnx",
        ),
        validate_and_save(
            scatter_max_invalid_zero(),
            SOUND / "task158_scatter_max_invalid_zero.onnx",
        ),
        validate_and_save(
            drop_unused_initializers(scatter_max_invalid_zero()),
            SOUND / "task158_scatter_max_invalid_zero_pruned.onnx",
        ),
        validate_and_save(
            scatter_max_orientation_only(),
            SOUND / "task158_scatter_max_orientation_only.onnx",
        ),
    ]
    (HERE / "evidence/exact_rewrite_build.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
