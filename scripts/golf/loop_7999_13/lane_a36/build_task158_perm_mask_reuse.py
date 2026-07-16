#!/usr/bin/env python3
"""Reuse task158's permutation masks for both scoring and q reordering."""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "scripts/golf/loop_8000_46/baseline_models/task158.onnx"
OUTPUT = HERE / "task158_perm_mask_reuse_7615.onnx"


def einsum(node: onnx.NodeProto, inputs: list[str], output: str, equation: str) -> None:
    node.op_type = "Einsum"
    del node.input[:]
    node.input.extend(inputs)
    del node.output[:]
    node.output.append(output)
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("equation", equation)])
    node.domain = ""
    node.name = output


def build() -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    (HERE / "baseline_task158.onnx").write_bytes(SOURCE.read_bytes())

    users = [
        (index, node.op_type, list(node.input))
        for index, node in enumerate(model.graph.node)
        if "more_perm_indices" in node.input
    ]
    assert users == [(119, "Gather", ["more_perm_indices", "best_perm"])]

    # The six [3,3] masks already encode exactly the same permutations as the
    # redundant [6,3] index bank.  Select the winning mask, then apply its
    # one-hot rows directly to q_row/q_col.  Each row has exactly one 1, so the
    # fp16 sums are selection-equivalent and introduce no numerical reduction.
    select = model.graph.node[119]
    select.input[0] = "more_perm_mask"
    select.output[0] = "selected_perm_mask"
    select.name = "selected_perm_mask"
    einsum(model.graph.node[120], ["selected_perm_mask", "q_row"], "q_row_m", "bij,bj->bi")
    einsum(model.graph.node[124], ["selected_perm_mask", "q_col"], "q_col_m", "bij,bj->bi")

    kept = [item for item in model.graph.initializer if item.name != "more_perm_indices"]
    assert len(kept) + 1 == len(model.graph.initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    vi = next(item for item in model.graph.value_info if item.name == "selected_perm")
    vi.name = "selected_perm_mask"
    vi.type.tensor_type.elem_type = TensorProto.FLOAT16
    del vi.type.tensor_type.shape.dim[:]
    for value in (1, 3, 3):
        vi.type.tensor_type.shape.dim.add().dim_value = value

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    inferred_vi = {
        item.name: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in inferred.graph.value_info
    }
    assert inferred_vi["selected_perm_mask"] == [1, 3, 3]
    assert inferred_vi["q_row_m"] == [1, 3]
    assert inferred_vi["q_col_m"] == [1, 3]
    return model


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(OUTPUT)
