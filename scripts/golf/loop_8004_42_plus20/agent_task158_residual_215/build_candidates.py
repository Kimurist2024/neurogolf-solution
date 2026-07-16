#!/usr/bin/env python3
"""Build complete-support exact residual task158 candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others/71407/task158.onnx"
EXPECTED_SHA = "127984c6807d84559bbf74fd58e3b09a66459d142cef65a8635647e64f5e59fd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def drop_unused_initializers(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(value.name for value in model.graph.output)
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def replace_role_threshold_chain_with_exact_bits(model: onnx.ModelProto) -> None:
    """Classify all reachable positive anchor scores by their low-role bits.

    Exhaustive enumeration of every legal endpoint-block local geometry gives
    low scores {2,8,10,24,26,72,106} and exactly doubled high scores
    {4,16,20,48,52,144,212}.  Bitmask 0b1010 is nonzero for every low score
    and zero for every high score (and zero).  Thus the existing validity test
    plus XOR reproduces ``anchor_high`` exactly without four threshold tensors.
    """
    removed = {
        "more_role_low",
        "more_role_mid",
        "more_role_u8",
        "role_threshold",
        "anchor_high",
        "low_mask",
    }
    nodes = list(model.graph.node)
    first = next(
        i for i, node in enumerate(nodes) if list(node.output) == ["more_role_low"]
    )
    anchor_valid_i = next(
        i for i, node in enumerate(nodes) if list(node.output) == ["anchor_valid"]
    )
    kept = [node for node in nodes if not any(out in removed for out in node.output)]
    first_kept = sum(
        1
        for node in nodes[:first]
        if not any(out in removed for out in node.output)
    )
    valid_kept = sum(
        1
        for node in nodes[: anchor_valid_i + 1]
        if not any(out in removed for out in node.output)
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray(10, dtype=np.uint8), "anchor_low_bit_mask"
        )
    )
    bit_nodes = [
        helper.make_node(
            "Cast", ["top_values"], ["anchor_score_u8"],
            to=onnx.TensorProto.UINT8, name="anchor_score_u8"
        ),
        helper.make_node(
            "BitwiseAnd",
            ["anchor_score_u8", "anchor_low_bit_mask"],
            ["anchor_low_bits"],
            name="anchor_low_bits",
        ),
        helper.make_node(
            "Greater",
            ["anchor_low_bits", "pq_u8_zero"],
            ["low_mask"],
            name="low_mask",
        ),
    ]
    kept[first_kept:first_kept] = bit_nodes
    # Account for the three nodes inserted before the retained anchor_valid.
    kept.insert(
        valid_kept + len(bit_nodes),
        helper.make_node(
            "Xor",
            ["anchor_valid", "low_mask"],
            ["anchor_high"],
            name="anchor_high",
        ),
    )
    del model.graph.node[:]
    model.graph.node.extend(kept)
    remove_value_info(
        model,
        {
            "more_role_low",
            "more_role_mid",
            "more_role_u8",
            "role_threshold",
        },
    )
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info(
                "anchor_score_u8", onnx.TensorProto.UINT8, [1, 8]
            ),
            helper.make_tensor_value_info(
                "anchor_low_bits", onnx.TensorProto.UINT8, [1, 8]
            ),
        ]
    )
    drop_unused_initializers(model)


def reuse_score_u8_for_exact_phase0(model: onnx.ModelProto) -> None:
    """Use the role classifier's cast for the first nested phase predicate.

    On the exhaustive support there is no score in [5, 7], hence incumbent
    ``score >= 6`` is exactly ``uint8(score) > 4``.  Scalar 4 already exists as
    ``lutnp_shift4``, removing the dedicated float16 cutoff without a tensor.
    """
    nodes = list(model.graph.node)
    cast_i = next(
        i for i, node in enumerate(nodes)
        if list(node.output) == ["anchor_score_u8"]
    )
    cast = nodes.pop(cast_i)
    phase_i = next(
        i for i, node in enumerate(nodes) if list(node.output) == ["phase_ge_0"]
    )
    phase = nodes[phase_i]
    assert phase.op_type == "GreaterOrEqual"
    assert list(phase.input) == ["top_values", "phase_cut_0"]
    phase.op_type = "Greater"
    del phase.input[:]
    phase.input.extend(["anchor_score_u8", "lutnp_shift4"])
    nodes.insert(phase_i, cast)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    drop_unused_initializers(model)


def validate(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    bad = []
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        shape = value.type.tensor_type.shape
        if any(
            not dim.HasField("dim_value") or dim.dim_value <= 0
            for dim in shape.dim
        ):
            bad.append(value.name)
    if bad:
        raise ValueError(f"non-static inferred values: {bad[:8]}")


def main() -> None:
    actual_sha = sha256(SOURCE)
    if actual_sha != EXPECTED_SHA:
        raise RuntimeError(f"task158 staged authority changed: {actual_sha}")
    for name in ("candidates", "evidence"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    model = onnx.load(SOURCE)
    replace_role_threshold_chain_with_exact_bits(model)
    reuse_score_u8_for_exact_phase0(model)
    validate(model)
    output = HERE / "candidates/task158_exact_anchor_role_bits.onnx"
    onnx.save(model, output)
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": actual_sha,
        "candidate": str(output.relative_to(ROOT)),
        "candidate_sha256": sha256(output),
        "candidate_bytes": output.stat().st_size,
        "rewrite": (
            "complete-support anchor-role threshold chain -> uint8 bitmask 0b1010"
        ),
    }
    (HERE / "evidence/build.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
