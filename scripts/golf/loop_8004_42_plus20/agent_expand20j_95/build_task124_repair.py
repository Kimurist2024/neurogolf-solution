#!/usr/bin/env python3
"""Build allocator-safe variants of the one-byte task124 Split shave."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"


def main() -> int:
    got = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if got != AUTHORITY_SHA256:
        raise SystemExit(f"authority drift: {got}")
    with zipfile.ZipFile(AUTHORITY) as archive:
        data = archive.read("task124.onnx")
    authority_sha = hashlib.sha256(data).hexdigest()
    outdir = HERE / "repair124"
    outdir.mkdir(exist_ok=True)
    rows = []
    for depth in (1, 2):
        model = onnx.load_model_from_string(data)
        split_index = next(i for i, node in enumerate(model.graph.node) if node.op_type == "Split")
        split = model.graph.node[split_index]
        if list(split.output) != ["r0", "r1", "r2", "r3", "r4"]:
            raise RuntimeError(f"unexpected Split outputs: {list(split.output)}")
        split.output[3] = ""
        source = split.input[0]
        identities = []
        for index in range(depth):
            target = f"row_codes_safe_{index + 1}"
            identities.append(helper.make_node("Identity", [source], [target], name=f"allocator_barrier_{index + 1}"))
            source = target
        split.input[0] = source
        for offset, node in enumerate(identities):
            model.graph.node.insert(split_index + offset, node)
        onnx.checker.check_model(model, full_check=True)
        payload = model.SerializeToString()
        path = outdir / f"task124_empty_r3_identity{depth}.onnx"
        path.write_bytes(payload)
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "serialized_bytes": len(payload),
                "identity_depth": depth,
            }
        )
    result = {
        "authority_zip_sha256": got,
        "authority_task124_sha256": authority_sha,
        "variants": rows,
    }
    (HERE / "audit" / "task124_repair_build.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
