#!/usr/bin/env python3
"""Extract exact 7999.13 A7 members and inventory their structure."""

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
TASKS = (174, 153, 325, 71, 55, 88, 86)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    assert digest(BASE.read_bytes()) == EXPECTED
    rows = []
    with zipfile.ZipFile(BASE) as archive:
        for task in TASKS:
            name = f"task{task:03d}.onnx"
            data = archive.read(name)
            path = HERE / "baseline" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            model = onnx.load_from_string(data)
            used = {x for node in model.graph.node for x in node.input if x}
            inits = []
            for init in model.graph.initializer:
                array = numpy_helper.to_array(init)
                inits.append({
                    "name": init.name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "elements": int(array.size),
                    "unique": int(np.unique(array).size),
                    "used": init.name in used,
                })
            rows.append({
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(data),
                "bytes": len(data),
                "nodes": len(model.graph.node),
                "ops": dict(Counter(node.op_type for node in model.graph.node)),
                "params": sum(item["elements"] for item in inits),
                "initializers": inits,
                "value_info": len(model.graph.value_info),
                "opset": [(item.domain, item.version) for item in model.opset_import],
            })
    out = {"baseline_zip_sha256": EXPECTED, "members": rows}
    (HERE / "baseline_manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
