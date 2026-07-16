#!/usr/bin/env python3
"""Remove task183's output-unreachable `hold_u8` Min exactly."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8018.91.zip"
OUTPUT = HERE / "candidates" / "task183_dead_min_removed.onnx"
ARCHIVE_SHA256 = "e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091"
MEMBER_SHA256 = "b2bc81fbe6bbd288b1e4f59048a38216530be07f3896bdfe2b4d568e64d5849b"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if sha256(AUTHORITY.read_bytes()) != ARCHIVE_SHA256:
        raise RuntimeError("8018.91 authority drift")
    with zipfile.ZipFile(AUTHORITY) as archive:
        source = archive.read("task183.onnx")
    if sha256(source) != MEMBER_SHA256:
        raise RuntimeError("task183 authority member drift")
    model = onnx.load_model_from_string(source)

    consumers = {name for node in model.graph.node for name in node.input if name}
    outputs = {value.name for value in model.graph.output}
    removed = []
    kept = []
    for index, node in enumerate(model.graph.node):
        if list(node.output) == ["hold_u8"] and "hold_u8" not in consumers and "hold_u8" not in outputs:
            removed.append({"index": index, "op_type": node.op_type, "output": "hold_u8"})
        else:
            kept.append(node)
    if removed != [{"index": 27, "op_type": "Min", "output": "hold_u8"}]:
        raise RuntimeError(f"unexpected dead-node proof: {removed}")
    del model.graph.node[:]
    model.graph.node.extend(kept)
    kept_value_info = [value for value in model.graph.value_info if value.name != "hold_u8"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    model.producer_name = "codex-task183-dead-min-removal"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    out_shape = [int(dim.dim_value) for dim in inferred.graph.output[0].type.tensor_type.shape.dim]
    if out_shape != [1, 10, 30, 30]:
        raise RuntimeError(f"unexpected output shape: {out_shape}")
    blob = model.SerializeToString()
    OUTPUT.write_bytes(blob)
    result = {
        "authority_member_sha256": MEMBER_SHA256,
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "candidate_sha256": sha256(blob),
        "removed": removed,
        "proof": "hold_u8 is neither a graph output nor an input of any node",
        "output_shape": out_shape,
    }
    (HERE / "task183_dead_min_build.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
