#!/usr/bin/env python3
"""Build fail-closed task158 repairs from the exact 8008.14 member."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8008.14.zip"
EXPECTED_SHA = "2823587ecc3f1b5b158357b5c32638003130f133ba6ab64a35337238f134aead"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def remove_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def remove_redundant_source_high(model: onnx.ModelProto) -> None:
    """Eliminate source_high_mask without introducing a false initializer.

    partial_low_mask is false at both source corners: at the low source corner
    low_mask XOR source_low_mask is false, and at the high source corner both
    inputs are false.  Therefore

      partial_high = Where(source_corner, partial_low, anchor_high)

    is exactly anchor_high with the source high corner removed.
    """
    source_high_index = next(
        i for i, node in enumerate(model.graph.node)
        if list(node.output) == ["source_high_mask"]
    )
    partial_high_index = next(
        i for i, node in enumerate(model.graph.node)
        if list(node.output) == ["partial_high_mask"]
    )
    source_high = model.graph.node[source_high_index]
    partial_high = model.graph.node[partial_high_index]
    assert source_high.op_type == "And"
    assert list(source_high.input) == ["source_corner", "anchor_high"]
    assert partial_high.op_type == "Xor"
    assert list(partial_high.input) == ["anchor_high", "source_high_mask"]

    nodes = list(model.graph.node)
    nodes[partial_high_index] = helper.make_node(
        "Where",
        ["source_corner", "partial_low_mask", "anchor_high"],
        ["partial_high_mask"],
        name="partial_high_mask",
    )
    del nodes[source_high_index]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    remove_value_info(model, {"source_high_mask"})


def gate_invalid_object_base(model: onnx.ModelProto) -> None:
    """Force invalid Scatter indices in range while preserving valid slots.

    The incumbent already gates invalid update codes to zero and uses
    ScatterElements(reduction=max), but still evaluates their indices.  Its
    arbitrary invalid pairings can exceed the 650-element seed.  Keep the
    valid base unchanged and force an invalid base to existing float16 scalar
    minus one.  The flattened local-offset LUT is at most 52 and magnitude is
    at most 12.5, so every invalid index is in [-1, 649], entirely inside
    ScatterElements' accepted inclusive range [-650, 649].
    """
    index = next(
        i for i, node in enumerate(model.graph.node)
        if list(node.output) == ["obj_base"]
    )
    node = model.graph.node[index]
    node.output[0] = "obj_base_ungated"
    node.name = "obj_base_ungated"
    gated = helper.make_node(
        "Where",
        ["p_valid", "obj_base_ungated", "neg_priority"],
        ["obj_base"],
        name="obj_base",
    )
    nodes = list(model.graph.node)
    nodes.insert(index + 1, gated)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.value_info.append(
        helper.make_tensor_value_info(
            "obj_base_ungated", onnx.TensorProto.FLOAT16, [1, 3]
        )
    )


def combine_anchor_affines(model: onnx.ModelProto) -> None:
    """Replace exact ``2*x + bit`` pairs by one variadic Sum output."""
    for axis in ("row", "col"):
        doubled = f"tile_{axis}2"
        output = f"anchor_{axis}s"
        base = f"tile_{axis}"
        phase = f"phase_{axis}"
        double_index = next(
            i for i, node in enumerate(model.graph.node)
            if list(node.output) == [doubled]
        )
        output_index = next(
            i for i, node in enumerate(model.graph.node)
            if list(node.output) == [output]
        )
        double_node = model.graph.node[double_index]
        output_node = model.graph.node[output_index]
        assert double_node.op_type == "Mul"
        assert list(double_node.input) == [base, "fc_two_f16"]
        assert output_node.op_type == "Add"
        assert list(output_node.input) == [doubled, phase]
        nodes = list(model.graph.node)
        nodes[double_index] = helper.make_node(
            "Sum", [base, base, phase], [output], name=output
        )
        del nodes[output_index]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        remove_value_info(model, {doubled})


def combine_integer_affines(model: onnx.ModelProto) -> None:
    """Fuse exact small-integer float16 multiply-add pairs into Sum.

    Every participating value is an integer or half-integer and every partial
    sum stays within the exact float16 lattice below 1024.  This is therefore
    algebraically and bitwise exact, independent of Sum association order.
    """
    specifications = (
        ("p_row_pack", "p_z", "p_row", "p_col", 32),
        ("q_row_pack", "q_z", "q_row", "q_col", 32),
        ("obj_top30", "obj_base", "obj_top", "obj_left", 25),
    )
    for product, output, base, addend, multiplier in specifications:
        product_index = next(
            i for i, node in enumerate(model.graph.node)
            if list(node.output) == [product]
        )
        output_index = next(
            i for i, node in enumerate(model.graph.node)
            if list(node.output) == [output]
        )
        product_node = model.graph.node[product_index]
        output_node = model.graph.node[output_index]
        assert product_node.op_type == "Mul"
        assert output_node.op_type == "Add"
        assert list(output_node.input) == [product, addend]
        nodes = list(model.graph.node)
        nodes[product_index] = helper.make_node(
            "Sum",
            [base] * multiplier + [addend],
            [output],
            name=output,
        )
        del nodes[output_index]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        remove_value_info(model, {product})


def combine_magnitude_shift(model: onnx.ModelProto) -> None:
    """Replace ``(i << 1) | i`` by exact uint8 ``3*i`` for i in {0,1,2}."""
    doubled_index = next(
        i for i, node in enumerate(model.graph.node)
        if list(node.output) == ["worker_safe_mag_index_x2"]
    )
    shift_index = next(
        i for i, node in enumerate(model.graph.node)
        if list(node.output) == ["worker_safe_mag_shift"]
    )
    doubled = model.graph.node[doubled_index]
    shift = model.graph.node[shift_index]
    assert doubled.op_type == "BitShift"
    assert shift.op_type == "BitwiseOr"
    nodes = list(model.graph.node)
    nodes[doubled_index] = helper.make_node(
        "Mul",
        ["worker_safe_mag_index_u8", "phase_u8_3"],
        ["worker_safe_mag_shift"],
        name="worker_safe_mag_shift",
    )
    del nodes[shift_index]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    remove_value_info(model, {"worker_safe_mag_index_x2"})


def drop_unused_initializers(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(value.name for value in model.graph.output)
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


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
    baseline_dir = HERE / "baseline"
    candidates_dir = HERE / "candidates"
    sound_dir = HERE / "sound"
    evidence_dir = HERE / "evidence"
    for directory in (baseline_dir, candidates_dir, sound_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(AUTHORITY) as archive:
        payload = archive.read("task158.onnx")
    actual_sha = sha256(payload)
    if actual_sha != EXPECTED_SHA:
        raise RuntimeError(f"authority task158 SHA changed: {actual_sha}")
    baseline_path = baseline_dir / "task158.onnx"
    baseline_path.write_bytes(payload)

    source_high_only = onnx.load_from_string(payload)
    remove_redundant_source_high(source_high_only)
    repair_only = onnx.load_from_string(payload)
    gate_invalid_object_base(repair_only)
    repaired = onnx.load_from_string(payload)
    remove_redundant_source_high(repaired)
    gate_invalid_object_base(repaired)
    repaired_sum = onnx.load_from_string(payload)
    combine_anchor_affines(repaired_sum)
    gate_invalid_object_base(repaired_sum)
    repaired_affine_sum = onnx.load_from_string(payload)
    combine_anchor_affines(repaired_affine_sum)
    combine_integer_affines(repaired_affine_sum)
    combine_magnitude_shift(repaired_affine_sum)
    gate_invalid_object_base(repaired_affine_sum)
    drop_unused_initializers(repaired_affine_sum)

    rows = [
        save(
            source_high_only,
            candidates_dir / "task158_source_high_only_unsafe_control.onnx",
        ),
        save(
            repair_only,
            candidates_dir / "task158_invalid_base_repair_control.onnx",
        ),
        save(
            repaired,
            candidates_dir / "task158_invalid_base_repair_source_high.onnx",
        ),
        save(
            repaired_sum,
            candidates_dir / "task158_invalid_base_repair_anchor_sum.onnx",
        ),
        save(
            repaired_affine_sum,
            candidates_dir / "task158_invalid_base_repair_affine_sum.onnx",
        ),
        save(
            repaired_affine_sum,
            sound_dir / "task158_exact_repair_cost7529.onnx",
        ),
    ]
    result = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_zip_sha256": sha256(AUTHORITY.read_bytes()),
        "member": "task158.onnx",
        "member_sha256": actual_sha,
        "rows": rows,
    }
    (evidence_dir / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
