#!/usr/bin/env python3
"""Extract and inventory exact A11 members and retain the task105 candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_7999.13.zip"
EXPECTED = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
SOURCE_105 = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task105_r01_static198.onnx"
TASKS = (65, 88, 105, 189, 224, 240)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def describe(path: Path, task: int, kind: str) -> dict[str, object]:
    data = path.read_bytes()
    model = onnx.load_from_string(data)
    initializers = []
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        initializers.append(
            {
                "name": init.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "unique": int(np.unique(array).size),
            }
        )
    return {
        "task": task,
        "kind": kind,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(data),
        "bytes": len(data),
        "nodes": len(model.graph.node),
        "ops": dict(Counter(node.op_type for node in model.graph.node)),
        "params": sum(item["elements"] for item in initializers),
        "initializers": initializers,
        "value_info": len(model.graph.value_info),
        "opset": [(item.domain, item.version) for item in model.opset_import],
    }


def main() -> None:
    if digest(BASE.read_bytes()) != EXPECTED:
        raise RuntimeError("baseline archive hash mismatch")
    rows = []
    with zipfile.ZipFile(BASE) as archive:
        for task in TASKS:
            name = f"task{task:03d}.onnx"
            data = archive.read(name)
            path = HERE / "baseline" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            rows.append(describe(path, task, "exact_baseline"))
    candidate = HERE / "candidate_task105_static198.onnx"
    shutil.copyfile(SOURCE_105, candidate)
    rows.append(describe(candidate, 105, "retained_candidate"))
    output = {"baseline_zip_sha256": EXPECTED, "models": rows}
    (HERE / "model_manifest.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
