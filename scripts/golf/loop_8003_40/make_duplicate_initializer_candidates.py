#!/usr/bin/env python3
"""Deduplicate byte-identical top-level ONNX initializers exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx


def tensor_key(tensor: onnx.TensorProto) -> bytes:
    clone = onnx.TensorProto()
    clone.CopyFrom(tensor)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite(source: Path, destination: Path) -> dict[str, object]:
    model = onnx.load(source, load_external_data=False)
    canonical: dict[bytes, str] = {}
    replacements: dict[str, str] = {}
    kept = []
    for initializer in model.graph.initializer:
        key = tensor_key(initializer)
        if key in canonical:
            replacements[initializer.name] = canonical[key]
        else:
            canonical[key] = initializer.name
            kept.append(initializer)

    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value in replacements:
                node.input[index] = replacements[value]

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "source_sha256": sha256(source),
        "candidate_sha256": sha256(destination),
        "replacements": replacements,
        "strict_checker": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = rewrite(args.source, args.destination)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
