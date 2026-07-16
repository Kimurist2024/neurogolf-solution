#!/usr/bin/env python3
"""Build mechanically exact candidates found during the 201..350 audit."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"


def bypass_identity(task: int, node_index: int) -> dict[str, object]:
    with zipfile.ZipFile(AUTHORITY) as archive:
        source_bytes = archive.read(f"task{task:03d}.onnx")
    model = onnx.load_model_from_string(source_bytes)
    node = model.graph.node[node_index]
    if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
        raise RuntimeError("requested node is not a unary Identity")
    source, target = node.input[0], node.output[0]
    del model.graph.node[node_index]
    for consumer in model.graph.node:
        for index, name in enumerate(consumer.input):
            if name == target:
                consumer.input[index] = source
    for output in model.graph.output:
        if output.name == target:
            raise RuntimeError("graph-output Identity bypass requires explicit rename")
    # Custom/contrib golf operators are not all understood by ONNX's shape
    # inferencer, so preserve the authority's explicit static value_info.
    kept_value_info = [value for value in model.graph.value_info if value.name != target]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    data = model.SerializeToString()
    digest = hashlib.sha256(data).hexdigest()
    try:
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        rejected = HERE / "rejected"
        rejected.mkdir(parents=True, exist_ok=True)
        path = rejected / f"task{task:03d}_identity_bypass_{digest[:12]}.onnx"
        path.write_bytes(data)
        return {
            "task": task,
            "transformation": f"remove Identity node {node_index}: {target} <- {source}",
            "authority_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "candidate_sha256": digest,
            "candidate": str(path.relative_to(ROOT)),
            "status": "rejected",
            "reason": f"static checker: {type(exc).__name__}: {exc}",
        }
    candidates = HERE / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    path = candidates / f"task{task:03d}_identity_bypass_{digest[:12]}.onnx"
    path.write_bytes(data)
    return {
        "task": task,
        "transformation": f"remove Identity node {node_index}: {target} <- {source}",
        "authority_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "candidate_sha256": digest,
        "candidate": str(path.relative_to(ROOT)),
        "status": "built",
    }


def main() -> None:
    rows = [bypass_identity(75, 7)]
    (HERE / "exact_builds.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
