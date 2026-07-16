#!/usr/bin/env python3
"""Record checker, strict-shape, and prohibited lookup-op evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
FILES = [
    HERE / "task162_base.onnx",
    HERE / "task162_reuse_bool.onnx",
    HERE / "task162_cse.onnx",
    HERE / "task268_base.onnx",
]
LOOKUP_OPS = {"TfIdfVectorizer"}


def audit(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    checker_ok = True
    checker_error = ""
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:  # pragma: no cover - evidence path
        checker_ok = False
        checker_error = repr(exc)

    strict_shape_ok = True
    strict_shape_error = ""
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:
        strict_shape_ok = False
        strict_shape_error = repr(exc)

    lookup_ops = sorted({node.op_type for node in model.graph.node if node.op_type in LOOKUP_OPS})
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checker_ok": checker_ok,
        "checker_error": checker_error,
        "strict_shape_ok": strict_shape_ok,
        "strict_shape_error": strict_shape_error,
        "lookup_ops": lookup_ops,
        "node_count": len(model.graph.node),
        "initializer_param_count": sum(
            max(1, int(__import__("math").prod(initializer.dims)))
            for initializer in model.graph.initializer
        ),
    }


def main() -> None:
    result = {"models": [audit(path) for path in FILES]}
    output = HERE / "strict_gate_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
