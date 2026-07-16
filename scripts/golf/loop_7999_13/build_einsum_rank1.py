#!/usr/bin/env python3
"""Factor selected near-rank-one 2-D Einsum initializers into two vectors."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


TARGETS = {36: "F", 234: "moment_weights_cloak", 256: "B", 348: "D", 350: "G", 356: "G"}


def get_equation(node: onnx.NodeProto) -> tuple[onnx.AttributeProto, str]:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr, attr.s.decode("ascii")
    raise RuntimeError("Einsum equation missing")


def factor_vectors(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    u, singular, vh = np.linalg.svd(array.astype(np.float64), full_matrices=False)
    scale = float(np.sqrt(singular[0]))
    left = (u[:, 0] * scale).astype(array.dtype)
    right = (vh[0, :] * scale).astype(array.dtype)
    error = float(np.max(np.abs(np.outer(left, right).astype(np.float64) - array)))
    return left, right, error


def build(model: onnx.ModelProto, initializer_name: str) -> tuple[onnx.ModelProto, dict[str, object]]:
    initializer = next(item for item in model.graph.initializer if item.name == initializer_name)
    array = np.asarray(numpy_helper.to_array(initializer))
    if array.ndim != 2:
        raise RuntimeError(f"{initializer_name} is not rank 2")
    left, right, error = factor_vectors(array)
    left_name = initializer_name + "__rank1_left"
    right_name = initializer_name + "__rank1_right"
    replacements = 0
    unsupported_uses: list[str] = []
    for node in model.graph.node:
        positions = [index for index, name in enumerate(node.input) if name == initializer_name]
        if not positions:
            continue
        if node.op_type != "Einsum":
            unsupported_uses.append(node.op_type)
            continue
        attr, equation = get_equation(node)
        if "->" not in equation:
            raise RuntimeError("explicit Einsum output required")
        lhs, rhs = equation.split("->", 1)
        operands = lhs.split(",")
        inputs = list(node.input)
        for position in reversed(positions):
            subscripts = operands[position]
            if len(subscripts) != 2 or subscripts[0] == subscripts[1]:
                raise RuntimeError(f"unsupported operand {subscripts}")
            inputs[position : position + 1] = [left_name, right_name]
            operands[position : position + 1] = [subscripts[0], subscripts[1]]
            replacements += 1
        del node.input[:]
        node.input.extend(inputs)
        attr.s = (",".join(operands) + "->" + rhs).encode("ascii")
    if unsupported_uses:
        raise RuntimeError(f"non-Einsum uses of {initializer_name}: {unsupported_uses}")
    if not replacements:
        raise RuntimeError(f"no uses replaced for {initializer_name}")
    kept = [item for item in model.graph.initializer if item.name != initializer_name]
    kept.extend(
        [numpy_helper.from_array(left, left_name), numpy_helper.from_array(right, right_name)]
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model)
    return model, {
        "initializer": initializer_name,
        "shape": list(array.shape),
        "replacements": replacements,
        "original_params": int(array.size),
        "candidate_params": int(left.size + right.size),
        "parameter_saving": int(array.size - left.size - right.size),
        "float32_max_abs_reconstruction_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(str(task) for task in TARGETS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in (int(item) for item in args.tasks.split(",") if item):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            candidate, details = build(model, TARGETS[task])
            output = args.output_dir / f"task{task:03d}.onnx"
            onnx.save(candidate, output)
            rows.append({"task": task, "path": str(output), **details})
    (args.output_dir / "build_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
