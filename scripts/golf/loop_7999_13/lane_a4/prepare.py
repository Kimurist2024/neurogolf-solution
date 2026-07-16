#!/usr/bin/env python3
"""Extract and inventory the exact 7999.13 incumbents for lane A4."""

from __future__ import annotations

import hashlib
import json
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
TASKS = (19, 34, 237, 250, 308, 324, 377)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if sha256(BASE.read_bytes()) != EXPECTED:
        raise RuntimeError("baseline identity changed")
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE) as archive:
        names = archive.namelist()
        for task in TASKS:
            member = f"task{task:03d}.onnx"
            payload = archive.read(member)
            path = HERE / "baseline" / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            model = onnx.load_from_string(payload)
            used = {name for node in model.graph.node for name in node.input if name}
            outputs = {name for node in model.graph.node for name in node.output if name}
            initializers = []
            for init in model.graph.initializer:
                array = numpy_helper.to_array(init)
                initializers.append(
                    {
                        "name": init.name,
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                        "elements": int(array.size),
                        "min": float(array.min()) if array.size else None,
                        "max": float(array.max()) if array.size else None,
                        "unique": int(np.unique(array).size),
                        "used": init.name in used,
                    }
                )
            records.append(
                {
                    "task": task,
                    "member_index": names.index(member),
                    "path": str(path.relative_to(ROOT)),
                    "bytes": len(payload),
                    "sha256": sha256(payload),
                    "nodes": len(model.graph.node),
                    "ops": dict(Counter(node.op_type for node in model.graph.node)),
                    "params": sum(item["elements"] for item in initializers),
                    "initializers": initializers,
                    "unused_initializers": [
                        item["name"] for item in initializers if not item["used"]
                    ],
                    "value_info": len(model.graph.value_info),
                    "orphan_node_outputs": sorted(outputs - used - {out.name for out in model.graph.output}),
                }
            )
    manifest = {"baseline_sha256": EXPECTED, "members": records}
    (HERE / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
