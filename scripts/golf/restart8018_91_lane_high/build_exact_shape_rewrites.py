#!/usr/bin/env python3
"""Build exact constant-shape and identity-reshape rewrites for lane high."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
AUTH = HERE / "authority"
OUT = HERE / "exact_shape_candidates"


def remove_output_node(model: onnx.ModelProto, output: str) -> None:
    matches = [node for node in model.graph.node if output in node.output]
    if len(matches) != 1:
        raise RuntimeError(f"{output}: expected one producer, got {len(matches)}")
    model.graph.node.remove(matches[0])


def add_i64(model: onnx.ModelProto, name: str, values: list[int]) -> None:
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)
    )


def replace_shape_with_i64(
    model: onnx.ModelProto, output: str, values: list[int]
) -> None:
    remove_output_node(model, output)
    add_i64(model, output, values)


def rewire(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def prune(model: onnx.ModelProto) -> None:
    used = {value for node in model.graph.node for value in node.input if value}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    # These golf graphs deliberately carry static value_info for custom ops.
    # Preserve live annotations, dropping only outputs of deleted nodes.
    live = {
        value for node in model.graph.node for value in node.output if value
    } | {value.name for value in model.graph.input} | {
        value.name for value in model.graph.output
    }
    kept_vi = [value for value in model.graph.value_info if value.name in live]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)


def build(task: int) -> tuple[Path, list[str]]:
    model = copy.deepcopy(onnx.load(str(AUTH / f"task{task:03d}.onnx")))
    changes: list[str] = []
    if task == 25:
        replace_shape_with_i64(model, "shape_1", [1])
        changes.append("Shape(input)[axis0] -> initializer [1]")
    elif task == 62:
        rewire(model, "row_idx_u8_r", "row_idx_u8")
        rewire(model, "col_idx_u8_r", "col_idx_u8")
        rewire(model, "pad_pads_i8_r", "pad_pads_i8")
        for output in (
            "idx_shape", "row_idx_u8_r", "col_idx_u8_r",
            "pad_shape", "pad_pads_i8_r",
        ):
            remove_output_node(model, output)
        changes.append("remove two identity Reshapes and shared Shape")
        changes.append("remove pad identity Reshape and Shape")
    elif task == 270:
        replace_shape_with_i64(model, "axis0_shape", [2])
        changes.append("Shape(Ridx8)[axis0] -> initializer [2]")
    elif task == 308:
        replace_shape_with_i64(model, "out_shape4_len", [4])
        replace_shape_with_i64(model, "topk_k_dyn", [4])
        changes.append("two invariant Shape outputs -> shared-valued initializers")
    elif task == 324:
        replace_shape_with_i64(model, "__f16sh", [1])
        remove_output_node(model, "top4_dyn")
        add_i64(model, "top4_dyn", [4])
        changes.append("Shape(input)[axis0] -> initializer [1]")
        changes.append("ConstantOfShape([1],4) -> initializer [4]")
    elif task == 374:
        replace_shape_with_i64(model, "__sp_shape", [1])
        changes.append("Shape(input)[axis0] -> initializer [1]")
    else:
        raise ValueError(task)
    prune(model)
    if task in {62, 308}:
        # These two rewrites infer cleanly from scratch; stale cloak annotations
        # refer to the deleted identity chain and must not be retained.
        del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"task{task:03d}_exact_shape.onnx"
    onnx.save(model, str(path))
    return path, changes


def main() -> None:
    rows = []
    for task in (25, 62, 270, 308, 324, 374):
        try:
            path, changes = build(task)
            rows.append({"task": task, "path": str(path), "changes": changes})
        except Exception as exc:  # noqa: BLE001
            rows.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
    (HERE / "exact_shape_build.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
