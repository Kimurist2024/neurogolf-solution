#!/usr/bin/env python3
"""Build semantics-preserving lane A4 algebraic candidates."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent


def producer_index(model: onnx.ModelProto, output: str) -> int:
    return next(i for i, node in enumerate(model.graph.node) if output in node.output)


def remove_stale_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def save(model: onnx.ModelProto, name: str) -> Path:
    onnx.checker.check_model(model, full_check=True)
    output = HERE / "candidates" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return output


def task377(row_eq: bool, color_eq: bool, color_prefix: bool) -> Path:
    model = onnx.load(HERE / "baseline" / "task377.onnx")
    removed: set[str] = set()

    if row_eq:
        # Integer-valued fp16 coordinates: (a >= b) * (b >= a) == (a == b).
        i1 = producer_index(model, "row_ge1")
        i2 = producer_index(model, "row_ge2")
        c1 = producer_index(model, "row_ge1_f")
        c2 = producer_index(model, "row_ge2_f")
        model.graph.node[i1].op_type = "Equal"
        del model.graph.node[i1].input[:]
        model.graph.node[i1].input.extend(["r30", "row_round"])
        for attr in list(model.graph.node[i1].attribute):
            model.graph.node[i1].attribute.remove(attr)
        line = model.graph.node[producer_index(model, "line_f")]
        line.input[1] = "row_ge1_f"
        line.input[2] = "row_ge1_f"
        # Delete later indices first.
        for index in sorted((i2, c2), reverse=True):
            del model.graph.node[index]
        removed.update(("row_ge2", "row_ge2_f"))

    if color_eq:
        i1 = producer_index(model, "c_ge1")
        i2 = producer_index(model, "c_ge2")
        c2 = producer_index(model, "c_ge2_f")
        model.graph.node[i1].op_type = "Equal"
        del model.graph.node[i1].input[:]
        model.graph.node[i1].input.extend(["color_ids", "colors5_f16"])
        for attr in list(model.graph.node[i1].attribute):
            model.graph.node[i1].attribute.remove(attr)
        final = model.graph.node[producer_index(model, "output")]
        final.input[1] = "c_ge1_f"
        for index in sorted((i2, c2), reverse=True):
            del model.graph.node[index]
        removed.update(("c_ge2", "c_ge2_f"))

    if color_prefix:
        # Cast the full coordinate vector once, then slice its exact 0..9 prefix.
        slice_index = producer_index(model, "color_ids_i16")
        cast_index = producer_index(model, "color_ids")
        model.graph.node[producer_index(model, "line")].input[1] = "coords_base"
        # same_count_u8 is intentionally UINT8, while coords_base is INT8.
        model.graph.node[producer_index(model, "same_count_u8")].input[1] = "two_f16"
        # Drop old int8 Slice and CastLike. Insert one fp16 Slice after r30 exists.
        for index in sorted((slice_index, cast_index), reverse=True):
            del model.graph.node[index]
        r30_index = producer_index(model, "r30")
        new_slice = onnx.helper.make_node(
            "Slice",
            ["r30", "starts0", "color_target", "starts0"],
            ["color_ids"],
        )
        model.graph.node.insert(r30_index + 1, new_slice)
        removed.add("color_ids_i16")

    remove_stale_value_info(model, removed)
    suffix = "_".join(
        label
        for enabled, label in (
            (row_eq, "roweq"),
            (color_eq, "coloreq"),
            (color_prefix, "prefix"),
        )
        if enabled
    )
    return save(model, f"task377_{suffix}.onnx")


def empty_unused_outputs(task: int, output_names: list[str]) -> Path | None:
    """Probe whether ORT/checker accept empty mandatory-but-unused outputs."""
    model = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
    removed = set(output_names)
    for name in output_names:
        node = model.graph.node[producer_index(model, name)]
        node.output[list(node.output).index(name)] = ""
    remove_stale_value_info(model, removed)
    try:
        return save(model, f"task{task:03d}_empty_unused.onnx")
    except Exception as exc:
        (HERE / f"task{task:03d}_empty_unused_rejected.txt").write_text(repr(exc) + "\n")
        return None


def task324_synthesize_quarter() -> Path:
    """Remove deg2 by synthesizing both 0.25 and the e=2 selector in Einsum.

    Selecting one 0.5 element from each of two existing ``base0`` operands
    gives 0.25.  ``refdiff[Z,A] * onehot[A] * seedsel[Z,B] * Emap[e,B]``
    is exactly the degree-two one-hot vector [0, 0, 1].  Both constructions
    stay inside the two existing Einsums and create no charged intermediates.
    """
    model = onnx.load(HERE / "baseline" / "task324.onnx")
    sig_sum = model.graph.node[producer_index(model, "sig_sum")]
    sig_sqsum = model.graph.node[producer_index(model, "sig_sqsum")]

    # Keep operands 0..8, replace deg2+unit2 with selected base0 factors.
    del sig_sum.input[:]
    sig_sum.input.extend(
        [
            "graph_input_cast_0",
            "active_oh",
            "seedsel",
            "coord",
            "coord",
            "binom",
            "signpow",
            "signpow",
            "refdiff",
            "base0",
            "onehot_values",
            "onehot_values",
            "base0",
            "onehot_values",
            "onehot_values",
        ]
    )
    for attr in sig_sum.attribute:
        if attr.name == "equation":
            attr.s = (
                b"narc,au,zu,pr,qc,epq,oq,be,Ab,iKjL,K,L,iMjN,M,N->iojz"
            )

    del sig_sqsum.input[:]
    sig_sqsum.input.extend(
        [
            "graph_input_cast_0",
            "active_oh",
            "seedsel",
            "coord",
            "coord",
            "binom",
            "signpow",
            "base0",
            "onehot_values",
            "onehot_values",
            "base0",
            "onehot_values",
            "onehot_values",
            "refdiff",
            "onehot_values",
            "seedsel",
            "Emap",
        ]
    )
    for attr in sig_sqsum.attribute:
        if attr.name == "equation":
            attr.s = (
                b"narc,au,zu,pr,qc,epq,oq,iKjL,K,L,iMjN,M,N,ZA,A,ZB,eB->iojz"
            )

    kept = [init for init in model.graph.initializer if init.name != "deg2"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return save(model, "task324_synth_quarter.onnx")


def main() -> None:
    outputs = []
    for row, color, prefix in (
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (True, True, True),
    ):
        outputs.append(str(task377(row, color, prefix)))
    outputs.append(str(task324_synthesize_quarter()))
    for task, names in ((19, ["ca0"]), (324, ["active_vals"]), (377, ["top_values"])):
        path = empty_unused_outputs(task, names)
        if path:
            outputs.append(str(path))
    (HERE / "candidate_build.json").write_text(
        json.dumps({"outputs": outputs}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
