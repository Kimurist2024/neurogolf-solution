#!/usr/bin/env python3
"""Replace task175's S initializer by an exact Msel factorization.

S = T @ Msel with the exact 3x3 row-swap permutation T.  Every S occurrence
in the one-node Einsum is replaced independently by T and the already-live
Msel.  Removing S (12 parameters) and adding T (9) saves three parameters on
top of the strict cost-134 W/V gauge candidate, with no scored node output.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "candidates" / "task175_gauge_remove_w_v.onnx"
OUTPUT = HERE / "candidates" / "task175_gauge_s_factor_reuse.onnx"
SOURCE_SHA256 = "acead77ce6b60ae5d5dd88e5c2c006cecdac6c9c5fd56bc97b56b37b72df8a1a"

sys.path.insert(0, str(ROOT / "scripts" / "golf" / "loop_7999_13"))
from build_factor_reuse_small import rewrite  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    source_blob = SOURCE.read_bytes()
    if sha256(source_blob) != SOURCE_SHA256:
        raise RuntimeError("cost-134 source drift")
    model = onnx.load_model_from_string(source_blob)
    change = rewrite(model, "S", "Msel", "right", "exact_global")
    if change["saving"] != 3 or change["max_abs_error"] != 0.0:
        raise RuntimeError(f"unexpected factor proof: {change}")
    model.producer_name = "codex-task175-gauge-s-factor-reuse"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    out_shape = [int(dim.dim_value) for dim in inferred.graph.output[0].type.tensor_type.shape.dim]
    if out_shape != [1, 10, 30, 30]:
        raise RuntimeError(f"unexpected output shape: {out_shape}")
    blob = model.SerializeToString()
    OUTPUT.write_bytes(blob)
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "candidate_sha256": sha256(blob),
        "factor_proof": change,
        "initializer_params": int(sum(np.prod(item.dims) for item in model.graph.initializer)),
        "output_shape": out_shape,
    }
    (HERE / "task175_s_factor_reuse_build.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
