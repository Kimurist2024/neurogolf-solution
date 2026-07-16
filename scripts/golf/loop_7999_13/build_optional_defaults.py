#!/usr/bin/env python3
"""Build exact-default optional-input candidates and remove newly dead constants."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def trim_trailing_empty(inputs: list[str]) -> list[str]:
    while inputs and not inputs[-1]:
        inputs.pop()
    return inputs


def build(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, object]]:
    initializers = {item.name: item for item in model.graph.initializer}
    changes: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        inputs = list(node.input)

        def zero_at(position: int) -> bool:
            if position >= len(inputs) or not inputs[position]:
                return False
            item = initializers.get(inputs[position])
            return item is not None and bool(np.all(numpy_helper.to_array(item) == 0))

        positions: list[int] = []
        if node.op_type in {"Conv", "ConvTranspose", "Gemm"} and zero_at(2):
            positions.append(2)
        if node.op_type == "QLinearConv" and zero_at(8):
            positions.append(8)
        if node.op_type in {"ConvInteger", "MatMulInteger"}:
            positions.extend(position for position in (2, 3) if zero_at(position))
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"} and zero_at(2):
            positions.append(2)
        if node.op_type == "Pad" and zero_at(2):
            positions.append(2)
        if node.op_type == "Slice" and len(inputs) >= 5 and inputs[4]:
            item = initializers.get(inputs[4])
            if item is not None and bool(np.all(numpy_helper.to_array(item) == 1)):
                positions.append(4)
        for position in sorted(set(positions)):
            changes.append(
                {
                    "node_index": index,
                    "op": node.op_type,
                    "input_index": position,
                    "initializer": inputs[position],
                }
            )
            inputs[position] = ""
        if positions:
            del node.input[:]
            node.input.extend(trim_trailing_empty(inputs))
    if not changes:
        raise RuntimeError("no optional defaults")
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    uses.update(item.name for item in model.graph.output)
    removed: list[dict[str, object]] = []
    kept = []
    for initializer in model.graph.initializer:
        if uses[initializer.name] == 0:
            elements = int(np.prod(initializer.dims, dtype=np.int64)) if initializer.dims else 1
            removed.append({"initializer": initializer.name, "elements": elements})
        else:
            kept.append(initializer)
    if not removed:
        raise RuntimeError("optional inputs changed but no parameter saving")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model, {
        "changes": changes,
        "removed": removed,
        "parameter_saving": sum(int(row["elements"]) for row in removed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            try:
                candidate, details = build(model)
            except Exception as error:
                if str(error) not in {
                    "no optional defaults", "optional inputs changed but no parameter saving"
                }:
                    errors.append({"task": task, "error": f"{type(error).__name__}: {error}"})
                continue
            output = args.output_dir / f"task{task:03d}.onnx"
            onnx.save(candidate, output)
            rows.append({"task": task, "path": str(output), **details})
    result = {"candidates": rows, "errors": errors}
    (args.output_dir / "build_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
