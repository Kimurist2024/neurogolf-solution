#!/usr/bin/env python3
"""Build byte-identical initializer-alias candidates from the 7999.13 ZIP."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def key(initializer: onnx.TensorProto) -> tuple[int, tuple[int, ...], bytes]:
    array = np.ascontiguousarray(numpy_helper.to_array(initializer))
    return int(initializer.data_type), tuple(int(x) for x in initializer.dims), array.tobytes()


def build(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    canonical: dict[tuple[int, tuple[int, ...], bytes], str] = {}
    replacements: dict[str, str] = {}
    removed_elements: dict[str, int] = {}
    for initializer in model.graph.initializer:
        tensor_key = key(initializer)
        if tensor_key in canonical:
            replacements[initializer.name] = canonical[tensor_key]
            removed_elements[initializer.name] = (
                int(np.prod(initializer.dims, dtype=np.int64)) if initializer.dims else 1
            )
        else:
            canonical[tensor_key] = initializer.name

    if not replacements:
        return model, []
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
    kept = [item for item in model.graph.initializer if item.name not in replacements]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    # Several incumbent models intentionally carry stale value_info shape hints;
    # preserve them byte-for-byte and require only the same structural checker
    # level here.  Runtime/fresh checks remain mandatory before acceptance.
    onnx.checker.check_model(model)
    changes = [
        {
            "removed": old,
            "replacement": replacements[old],
            "elements": removed_elements[old],
        }
        for old in sorted(replacements)
    ]
    return model, changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="18,77,107,173,219,233,366")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in (int(item) for item in args.tasks.split(",") if item):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            candidate, changes = build(model)
            output = args.output_dir / f"task{task:03d}.onnx"
            onnx.save(candidate, output)
            rows.append({"task": task, "path": str(output), "changes": changes})
    manifest = args.output_dir / "build_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
