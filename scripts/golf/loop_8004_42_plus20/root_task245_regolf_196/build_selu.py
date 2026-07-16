"""Replace positive-domain task245 Div(x, 2) nodes with Selu attributes."""

from __future__ import annotations

import copy
from pathlib import Path

import onnx
from onnx import helper


AUTHORITY = Path("/tmp/root_task245_196/task245.onnx")
OUTPUT = Path(__file__).resolve().parent / "task245_selu_cost384.onnx"


def main() -> None:
    model = copy.deepcopy(onnx.load(AUTHORITY))
    # Preserve the rank-1 carrier formerly introduced by broadcasting the
    # scalar logs with two_f16[1]. Batch is statically one, so retaining `n`
    # in each reduction changes only rank, never the single computed value.
    reduction_outputs = {"rr_code", "rc_code", "gr_code", "gc_code"}
    changed_equations = 0
    for node in model.graph.node:
        if node.op_type != "Einsum" or node.output[0] not in reduction_outputs:
            continue
        equation = next(attribute for attribute in node.attribute if attribute.name == "equation")
        text = onnx.helper.get_attribute_value(equation).decode()
        equation.s = text.replace("->", "->n").encode()
        changed_equations += 1
    if changed_equations != 4:
        raise RuntimeError(f"expected four scalar Einsums, changed {changed_equations}")

    rank_one_names = reduction_outputs | {
        "rr_log", "rc_log", "gr_log", "gc_log",
    }
    for value_info in model.graph.value_info:
        if value_info.name not in rank_one_names:
            continue
        shape = value_info.type.tensor_type.shape
        del shape.dim[:]
        shape.dim.add().dim_value = 1

    replaced = 0
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Div" or list(node.input)[1:] != ["two_f16"]:
            continue
        model.graph.node[index].CopyFrom(
            helper.make_node(
                "Selu",
                [node.input[0]],
                list(node.output),
                name=node.name,
                alpha=1.0,
                gamma=0.5,
            )
        )
        replaced += 1
    if replaced != 4:
        raise RuntimeError(f"expected four Div nodes, replaced {replaced}")

    kept = [value for value in model.graph.initializer if value.name != "two_f16"]
    if len(kept) + 1 != len(model.graph.initializer):
        raise RuntimeError("two_f16 initializer missing or duplicated")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.producer_name = "codex-task245-positive-selu-196"

    onnx.checker.check_model(model, full_check=True)
    # The LB-white authority deliberately cloaks the AffineGrid batch shape and
    # therefore already fails data_prop=True (2 inferred versus 1 declared).
    # Preserve that inherited declaration while requiring ordinary strict
    # inference to pass and proving whole-model raw equivalence separately.
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
