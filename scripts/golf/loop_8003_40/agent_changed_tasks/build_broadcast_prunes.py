#!/usr/bin/env python3
"""Build one-axis singleton-broadcast probes for the two one-node Einsum nets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "base_models"
OUT = HERE / "broadcast_prunes"
TARGETS = {
    260: ("E", "W", "Flip", "gate", "S3", "T3", "v1"),
    359: ("E", "M"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(task: int, name: str, axis: int, index: int) -> dict[str, object]:
    source = BASE / f"task{task:03d}.onnx"
    model = onnx.load(source)
    initializer = next(init for init in model.graph.initializer if init.name == name)
    original = numpy_helper.to_array(initializer)
    replacement_array = np.take(original, [index], axis=axis)
    replacement = numpy_helper.from_array(replacement_array, name=name)
    for init_index, init in enumerate(model.graph.initializer):
        if init.name == name:
            model.graph.initializer[init_index].CopyFrom(replacement)
            break
    tag = f"task{task:03d}_{name}_axis{axis}_idx{index}"
    row: dict[str, object] = {
        "task": task,
        "initializer": name,
        "axis": axis,
        "index": index,
        "source_shape": list(original.shape),
        "candidate_shape": list(replacement_array.shape),
        "source_elements": int(original.size),
        "candidate_elements": int(replacement_array.size),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        OUT.mkdir(parents=True, exist_ok=True)
        destination = OUT / f"{tag}.onnx"
        onnx.save(model, destination)
        row.update({
            "candidate": str(destination),
            "sha256": sha(destination),
            "checker": "PASS",
            "strict_shape_inference": "PASS",
        })
    except Exception as exc:
        row["build_error"] = repr(exc)
    return row


def main() -> None:
    rows = []
    for task, names in TARGETS.items():
        model = onnx.load(BASE / f"task{task:03d}.onnx")
        arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        for name in names:
            array = arrays[name]
            for axis, size in enumerate(array.shape):
                if size <= 1:
                    continue
                for index in sorted({0, size - 1}):
                    rows.append(build(task, name, axis, index))
    path = HERE / "broadcast_prune_build.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"built={sum('candidate' in row for row in rows)} total={len(rows)}")


if __name__ == "__main__":
    main()
