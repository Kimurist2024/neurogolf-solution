#!/usr/bin/env python3
"""Absorb task175's row-permuted Msel initializer into TA exactly.

Starting from the already strict-gated cost-134 W/V gauge rewrite, Msel is
exactly P@S for a 3x3 permutation P.  The only Msel occurrence contracts its
first axis with TA's first axis, so

    sum_P Msel[P,v] TA[P,l]
      = sum_P S[P,v] (P.T @ TA)[P,l].

We therefore alias that operand to the already-live S initializer, update TA,
and delete all 12 Msel parameters.  The one-node graph and equation are
otherwise unchanged; no scored activation is introduced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "candidates" / "task175_gauge_remove_w_v.onnx"
OUTPUT = HERE / "candidates" / "task175_gauge_msel_absorb.onnx"
SOURCE_SHA256 = "acead77ce6b60ae5d5dd88e5c2c006cecdac6c9c5fd56bc97b56b37b72df8a1a"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    source_blob = SOURCE.read_bytes()
    if sha256(source_blob) != SOURCE_SHA256:
        raise RuntimeError("cost-134 source drift")
    model = onnx.load_model_from_string(source_blob)
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        raise RuntimeError("unexpected task175 graph")
    arrays = {item.name: np.asarray(numpy_helper.to_array(item))
              for item in model.graph.initializer}
    if set(arrays) != {"Q", "S", "R", "Msel", "TA", "TB"}:
        raise RuntimeError(f"unexpected initializer set: {sorted(arrays)}")

    s = arrays["S"].astype(np.float32)
    msel = arrays["Msel"].astype(np.float32)
    ta = arrays["TA"].astype(np.float32)
    permutation = (msel @ s.T).astype(np.float32)
    if not np.array_equal(permutation @ s, msel):
        raise RuntimeError("Msel is not exactly reconstructed by P@S")
    if not np.array_equal(permutation @ permutation.T, np.eye(3, dtype=np.float32)):
        raise RuntimeError("Msel/S bridge is not an exact permutation")
    ta_new = (permutation.T @ ta).astype(np.float32)

    node = model.graph.node[0]
    replaced = 0
    for index, name in enumerate(node.input):
        if name == "Msel":
            node.input[index] = "S"
            replaced += 1
    if replaced != 1:
        raise RuntimeError(f"expected one Msel occurrence, found {replaced}")

    kept = []
    for item in model.graph.initializer:
        if item.name == "Msel":
            continue
        if item.name == "TA":
            kept.append(numpy_helper.from_array(ta_new, "TA"))
        else:
            kept.append(item)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.producer_name = "codex-task175-gauge-msel-absorb"
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
        "proof": {
            "Msel_equals_P_times_S": True,
            "P_is_permutation": True,
            "TA_rewrite": "TA_new=P.T@TA",
            "Msel_occurrences_replaced": replaced,
            "removed_parameters": int(msel.size),
            "added_parameters": 0,
        },
        "initializer_params": int(sum(np.prod(item.dims) for item in model.graph.initializer)),
        "output_shape": out_shape,
    }
    (HERE / "task175_msel_absorb_build.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
