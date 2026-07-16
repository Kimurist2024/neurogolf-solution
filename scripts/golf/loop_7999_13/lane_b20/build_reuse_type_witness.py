#!/usr/bin/env python3
"""Remove task162's redundant bool type-witness initializer.

CastLike reads only the element type of its second input.  ``dilge_107`` is an
already-computed bool tensor, so it is an exact replacement for scalar bool
initializer ``btmpl`` without adding a node.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "task162_base.onnx"
OUTPUT = HERE / "task162_reuse_bool.onnx"
MANIFEST = HERE / "task162_reuse_bool_build.json"


def main() -> None:
    model = onnx.load(SOURCE)
    final_cast = model.graph.node[-2]
    assert final_cast.op_type == "CastLike"
    assert list(final_cast.input) == ["dilf_108", "btmpl"]
    bool_outputs = {
        output
        for node in model.graph.node
        if node.op_type in {"Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal"}
        for output in node.output
    }
    assert "dilge_107" in bool_outputs
    final_cast.input[1] = "dilge_107"

    kept = [initializer for initializer in model.graph.initializer if initializer.name != "btmpl"]
    assert len(kept) + 1 == len(model.graph.initializer)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT)

    payload = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "change": "CastLike witness btmpl -> existing bool dilge_107; drop btmpl",
        "formal_equivalence": "CastLike uses only the element type of its second input; both are bool",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
