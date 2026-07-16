#!/usr/bin/env python3
"""Apply each conservative offline ONNX optimizer pass to the C2 task367 base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
import onnxoptimizer


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "base" / "task367.onnx"
OUT = HERE / "task367_optimizer_variants"


def digest(model: onnx.ModelProto) -> str:
    return hashlib.sha256(model.SerializeToString()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = onnx.load(SOURCE)
    base_digest = digest(base)
    rows = []
    for name in onnxoptimizer.get_available_passes():
        if name in {"rename_input_output", "rewrite_input_dtype"}:
            continue
        try:
            candidate = onnxoptimizer.optimize(base, [name], fixed_point=False)
            onnx.checker.check_model(candidate, full_check=True)
            onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
            candidate_digest = digest(candidate)
            changed = candidate_digest != base_digest
            path = OUT / f"{name}.onnx"
            if changed:
                onnx.save(candidate, path)
            rows.append(
                {
                    "pass": name,
                    "changed": changed,
                    "sha256": candidate_digest,
                    "nodes": len(candidate.graph.node),
                    "initializers": len(candidate.graph.initializer),
                    "path": str(path) if changed else None,
                }
            )
        except Exception as exc:
            rows.append({"pass": name, "error": f"{type(exc).__name__}: {exc}"})
    (HERE / "task367_optimizer_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(json.dumps([row for row in rows if row.get("changed")], indent=2))


if __name__ == "__main__":
    main()
