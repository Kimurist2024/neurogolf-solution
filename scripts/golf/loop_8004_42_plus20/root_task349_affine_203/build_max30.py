#!/usr/bin/env python3
"""Compound task349 shave: affine table removal plus max30 equality reuse."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "task349_affine_no_scalar.onnx"
OUTPUT = HERE / "task349_affine_max29.onnx"


def profile(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def main() -> None:
    model = onnx.load(SOURCE)
    uses = 0
    for node in model.graph.node:
        if node.output and node.output[0] in {"halo_end_is30", "beam_end_is30"}:
            assert node.op_type == "Equal"
            assert node.input[1] == "max30_i8"
            node.op_type = "Greater"
            node.input[1] = "max29_i8"
            uses += 1
    assert uses == 2
    assert sum("max30_i8" in node.input for node in model.graph.node) == 0
    kept = [x for x in model.graph.initializer if x.name != "max30_i8"]
    assert len(kept) + 1 == len(model.graph.initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)
    result = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "support_identity": "for valid task349 inputs, side and clipped halo_end are integer int8 values <=30; Equal(x,30) == Greater(x,29)",
        "source_profile": profile(SOURCE),
        "candidate_profile": profile(OUTPUT),
    }
    (HERE / "build_max30.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
