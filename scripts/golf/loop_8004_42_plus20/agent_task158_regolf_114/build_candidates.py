#!/usr/bin/env python3
"""Build strict-lower exact regolf candidates from the accepted cost-7529 graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task158_current_108/sound/"
    "task158_exact_repair_cost7529.onnx"
)
EXPECTED_SHA = "9d9a3ca8fb39856125925ea464ed1cc80f0301bd785ff7b60da37bd1c2b6b9d1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def set_value_info_type(
    model: onnx.ModelProto, name: str, elem_type: int, shape: list[int]
) -> None:
    values = [value for value in model.graph.value_info if value.name != name]
    del model.graph.value_info[:]
    model.graph.value_info.extend(values)
    model.graph.value_info.append(
        helper.make_tensor_value_info(name, elem_type, shape)
    )


def compact_masked_topk_ranks(model: onnx.ModelProto) -> None:
    """Replace float16 payload scores by exact uint8 positional ranks.

    ``top_values`` is already sorted descending.  Both later TopK nodes only
    compact selected slots while preserving that order; no consumer uses their
    Values outputs.  Fixed ranks 8..1 preserve every valid ordering exactly.
    Invalid slots tie at zero, but their gathered validity is false, assignment
    edges are invalid-cost, update codes are zero, and their bases are safely
    gated, so their internal tie order is unobservable.
    """
    rank_name = "top_positional_rank_u8"
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=np.uint8),
            rank_name,
        )
    )
    for prefix, mask in (
        ("low", "partial_low_mask"),
        ("high", "partial_high_mask"),
    ):
        score_name = f"{prefix}_rank_score"
        values_name = f"{prefix}_rank_values"
        node = next(
            node for node in model.graph.node
            if list(node.output) == [score_name]
        )
        assert node.op_type == "Where"
        assert list(node.input) == [mask, "top_values", "neg_priority"]
        del node.input[:]
        node.input.extend([mask, rank_name, "pq_u8_zero"])
        set_value_info_type(
            model, score_name, onnx.TensorProto.UINT8, [1, 8]
        )
        set_value_info_type(
            model, values_name, onnx.TensorProto.UINT8, [1, 3]
        )


def replace_initializer(
    model: onnx.ModelProto, name: str, array: np.ndarray
) -> None:
    index = next(
        i for i, item in enumerate(model.graph.initializer) if item.name == name
    )
    normalized = np.asarray(array)
    if normalized.ndim:
        normalized = np.ascontiguousarray(normalized)
    model.graph.initializer[index].CopyFrom(
        numpy_helper.from_array(normalized, name)
    )


def drop_unused_initializers(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(value.name for value in model.graph.output)
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def shift_out_anchor_bias(model: onnx.ModelProto) -> None:
    """Remove uniform +146 Conv bias by translating all score thresholds.

    Dynamic anchor weights and one-hot inputs are integral, so the bias-free
    Conv sum is integral and exactly represented around every decision boundary.
    All rank ordering is translation invariant.  The strict old ``>146`` floor
    becomes exact ``>=1`` and reuses ``fc_one_f16``.
    """
    conv = next(
        node for node in model.graph.node
        if list(node.output) == ["anchor_score"]
    )
    assert conv.op_type == "Conv"
    assert list(conv.input) == ["input", "anchor_weight", "da_anchor_bias"]
    del conv.input[2]

    for name in ("phase_cut_0", "phase_cut_1", "phase_cut_2"):
        item = next(x for x in model.graph.initializer if x.name == name)
        array = numpy_helper.to_array(item).copy()
        replace_initializer(model, name, (array - np.float16(146)).astype(np.float16))
    for name in ("more_role_0", "more_role_1", "more_role_2", "more_role_3"):
        item = next(x for x in model.graph.initializer if x.name == name)
        array = numpy_helper.to_array(item).astype(np.int16)
        shifted = (array - 146).astype(np.uint8)
        replace_initializer(model, name, shifted)

    valid = next(
        node for node in model.graph.node
        if list(node.output) == ["anchor_valid"]
    )
    assert valid.op_type == "Greater"
    assert list(valid.input) == ["top_values", "anchor_floor"]
    valid.op_type = "GreaterOrEqual"
    valid.input[1] = "fc_one_f16"
    drop_unused_initializers(model)


def alias_shifted_initializers(model: onnx.ModelProto) -> None:
    """CSE the uint8 scalar 2 exposed by translating more_role_0."""
    arrays = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }
    assert arrays["more_role_0"].dtype == arrays["phase_u8_2"].dtype
    assert np.array_equal(arrays["more_role_0"], arrays["phase_u8_2"])
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "more_role_0":
                node.input[index] = "phase_u8_2"
    drop_unused_initializers(model)


def scale_shifted_anchor_by_two(model: onnx.ModelProto) -> None:
    """Multiply the translated integral anchor score system by exact two."""
    for name in (
        "anchor_stencil",
        "phase_cut_0",
        "phase_cut_1",
        "phase_cut_2",
    ):
        item = next(x for x in model.graph.initializer if x.name == name)
        array = numpy_helper.to_array(item)
        replace_initializer(model, name, (array * 2).astype(array.dtype))
    for name in ("more_role_0", "more_role_1", "more_role_2", "more_role_3"):
        item = next(x for x in model.graph.initializer if x.name == name)
        array = numpy_helper.to_array(item).astype(np.uint16)
        doubled = array * 2
        if np.any(doubled > 255):
            raise ValueError(f"uint8 role overflow: {name} {doubled}")
        replace_initializer(model, name, doubled.astype(np.uint8))
    valid = next(
        node for node in model.graph.node
        if list(node.output) == ["anchor_valid"]
    )
    assert valid.op_type == "GreaterOrEqual"
    assert list(valid.input) == ["top_values", "fc_one_f16"]
    valid.input[1] = "fc_two_f16"


def alias_scaled_initializers(model: onnx.ModelProto) -> None:
    """CSE aliases exposed only after the exact x2 score scaling."""
    arrays = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }
    aliases = {
        "more_role_0": "lutnp_shift4",
        "phase_cut_2": "nm_d62",
    }
    for old, canonical in aliases.items():
        assert arrays[old].dtype == arrays[canonical].dtype
        assert arrays[old].shape == arrays[canonical].shape
        assert np.array_equal(arrays[old], arrays[canonical])
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                if name == old:
                    node.input[index] = canonical
    drop_unused_initializers(model)


def validate(model: onnx.ModelProto) -> None:
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


def save(model: onnx.ModelProto, path: Path) -> dict[str, object]:
    validate(model)
    onnx.save(model, path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path.read_bytes()),
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
    }


def main() -> None:
    payload = SOURCE.read_bytes()
    actual_sha = sha256(payload)
    if actual_sha != EXPECTED_SHA:
        raise RuntimeError(f"cost7529 authority changed: {actual_sha}")
    for name in ("baseline", "candidates", "sound", "evidence"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    (HERE / "baseline/task158.onnx").write_bytes(payload)

    rank = onnx.load_from_string(payload)
    compact_masked_topk_ranks(rank)
    shifted = onnx.load_from_string(payload)
    shift_out_anchor_bias(shifted)
    shifted_alias = onnx.load_from_string(payload)
    shift_out_anchor_bias(shifted_alias)
    alias_shifted_initializers(shifted_alias)
    shifted_scaled_alias = onnx.load_from_string(payload)
    shift_out_anchor_bias(shifted_scaled_alias)
    scale_shifted_anchor_by_two(shifted_scaled_alias)
    alias_scaled_initializers(shifted_scaled_alias)
    rows = [
        save(rank, HERE / "candidates/task158_uint8_rank_topk.onnx"),
        save(shifted, HERE / "candidates/task158_anchor_bias_shift.onnx"),
        save(shifted_alias, HERE / "candidates/task158_anchor_bias_shift_alias.onnx"),
        save(
            shifted_scaled_alias,
            HERE / "candidates/task158_anchor_bias_shift_scaled_alias.onnx",
        ),
        save(shifted_scaled_alias, HERE / "sound/task158_exact_regolf.onnx"),
    ]
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": actual_sha,
        "rows": rows,
    }
    (HERE / "evidence/build.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
