#!/usr/bin/env python3
"""Deduplicate two byte-identical scalar initializers in task233."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "baseline_models" / "task233.onnx"
OUT_DIR = ROOT / "lane_exact_initializer_dedup"
OUTPUT = OUT_DIR / "task233_exact_dedup.onnx"


def tensor_key(tensor: onnx.TensorProto) -> tuple[object, ...]:
    return (
        tensor.data_type,
        tuple(tensor.dims),
        bytes(tensor.raw_data),
        tuple(tensor.float_data),
        tuple(tensor.int32_data),
        tuple(tensor.int64_data),
        tuple(tensor.double_data),
        tuple(tensor.string_data),
    )


def parameter_count(model: onnx.ModelProto) -> int:
    return sum(max(1, int(np.prod(item.dims))) for item in model.graph.initializer)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = onnx.load(SOURCE, load_external_data=False)
    replacements = {
        "three_s16": "three_i16",
        "audit_one_i16": "one_i8",
    }

    by_name = {item.name: item for item in model.graph.initializer}
    for old_name, new_name in replacements.items():
        if tensor_key(by_name[old_name]) != tensor_key(by_name[new_name]):
            raise ValueError(f"initializer mismatch: {old_name} != {new_name}")

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]

    kept = [item for item in model.graph.initializer if item.name not in replacements]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT)

    source_model = onnx.load(SOURCE, load_external_data=False)
    report = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "source_params": parameter_count(source_model),
        "output_params": parameter_count(model),
        "replacements": replacements,
    }
    (OUT_DIR / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
