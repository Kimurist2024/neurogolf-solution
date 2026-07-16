#!/usr/bin/env python3
"""Build exact, non-promoted diagnostics inside this lane only."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import onnx
from onnx import shape_inference

HERE = Path(__file__).resolve().parent
BASE = HERE / "base"
CANDIDATES = HERE / "candidates"


def replace_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new


def main() -> int:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    source = BASE / "task096.onnx"
    model = onnx.load(source)
    identities = [
        (index, node)
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Identity"
    ]
    if len(identities) != 1:
        raise RuntimeError(f"expected one Identity, got {len(identities)}")
    index, node = identities[0]
    old, new = node.output[0], node.input[0]
    replace_uses(model, old, new)
    del model.graph.node[index]
    kept = [value for value in model.graph.value_info if value.name != old]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    target = CANDIDATES / "task096_identity_pruned_exact.onnx"
    onnx.save(model, target)
    structure = {"checker_full": False, "strict_data_prop": False}
    try:
        onnx.checker.check_model(model, full_check=True)
        structure["checker_full"] = True
    except Exception as exc:
        structure["checker_error"] = f"{type(exc).__name__}: {exc}"
    try:
        shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        structure["strict_data_prop"] = True
    except Exception as exc:
        structure["strict_error"] = f"{type(exc).__name__}: {exc}"
    proof = {
        "task": 96,
        "source": str(source),
        "candidate": str(target),
        "rewrite": {
            "node_index": index,
            "op": "Identity",
            "input": new,
            "output": old,
            "proof": "Identity(x)=x for every tensor; every consumer was rewired to x.",
        },
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "structure": structure,
    }
    (HERE / "diagnostic_build.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
