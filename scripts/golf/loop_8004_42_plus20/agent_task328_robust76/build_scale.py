#!/usr/bin/env python3
"""Scale task328's output-only Rflip factor by an exact positive power of two."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/root_sweep29/reuse_contract/task328_r001.onnx"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exponent", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.exponent <= 120:
        raise ValueError("exponent must stay within the finite float32 range")

    model = onnx.load(SOURCE)
    uses = {item.name: 0 for item in model.graph.initializer}
    for node in model.graph.node:
        for name in node.input:
            if name in uses:
                uses[name] += 1
    if uses.get("Rflip") != 1:
        raise RuntimeError(f"Rflip use count changed: {uses.get('Rflip')}")
    final = model.graph.node[-1]
    if final.op_type != "Einsum" or "Rflip" not in final.input or final.output != ["output"]:
        raise RuntimeError("Rflip is no longer an output-only final-Einsum operand")

    scale = np.float32(2.0**args.exponent)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("nonfinite scale")
    for index, item in enumerate(model.graph.initializer):
        if item.name != "Rflip":
            continue
        before = numpy_helper.to_array(item).astype(np.float32)
        after = before * scale
        if not np.isfinite(after).all():
            raise ValueError("scaled Rflip is nonfinite")
        replacement = numpy_helper.from_array(after, name="Rflip")
        model.graph.initializer[index].CopyFrom(replacement)
        break
    else:
        raise RuntimeError("Rflip initializer missing")

    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    data = args.output.read_bytes()
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output": str(args.output.resolve().relative_to(ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "exponent": args.exponent,
        "scale": float(scale),
        "changed_initializer": "Rflip",
        "initializer_use_count": uses["Rflip"],
        "output_only_final_einsum_factor": True,
        "serialized_bytes": len(data),
    }
    args.output.with_suffix(".build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
