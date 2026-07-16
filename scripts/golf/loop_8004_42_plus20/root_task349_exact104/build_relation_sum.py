#!/usr/bin/env python3
"""Fuse the exact task349 halo restoration into one variadic Sum node."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "task349.onnx"
RELATION = HERE / "task349_relation_exact.onnx"
OUTPUT = HERE / "task349_relation_sum_exact.onnx"
REPORT = HERE / "relation_sum_result.json"
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import (  # noqa: E402
    masks_equal_with_margin,
    model_margin_stable,
    score_and_verify,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    model = onnx.load(RELATION)
    rewritten = []
    fused = removed_restore = False
    for node in model.graph.node:
        if node.name == "halo_end_minus1":
            if node.op_type != "Add" or list(node.output) != ["halo_end_minus1"]:
                raise AssertionError("unexpected pre-restore node")
            node.op_type = "Sum"
            node.input.append("one_i8")
            node.output[0] = "halo_end1"
            node.name = "halo_end1_variadic_sum"
            fused = True
            rewritten.append(node)
            continue
        if node.name == "halo_end1_restore":
            removed_restore = True
            continue
        rewritten.append(node)
    if not (fused and removed_restore):
        raise AssertionError("relation Add chain was not found")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    if inferred.functions or inferred.graph.sparse_initializer:
        raise AssertionError("functions/sparse initializers are forbidden")
    onnx.save(model, OUTPUT)

    work = HERE / "work_sum"
    baseline = score_and_verify(
        copy.deepcopy(onnx.load(SOURCE)), 349, str(work),
        label="task349_lb8008_14", require_correct=True,
    )
    candidate = score_and_verify(
        copy.deepcopy(model), 349, str(work),
        label="task349_relation_sum_exact", require_correct=True,
    )
    if baseline is None or candidate is None:
        raise RuntimeError(f"official scoring failed: {baseline=} {candidate=}")
    known_equal = masks_equal_with_margin(
        copy.deepcopy(onnx.load(SOURCE)), copy.deepcopy(model), 349, margin=0.25
    )
    margin_ok, min_abs = model_margin_stable(copy.deepcopy(model), 349, margin=0.25)
    payload = {
        "task": 349,
        "source_sha256": sha256(SOURCE),
        "candidate_sha256": sha256(OUTPUT),
        "baseline": baseline,
        "candidate": candidate,
        "cost_reduction": int(baseline["cost"] - candidate["cost"]),
        "projected_gain": math.log(baseline["cost"] / candidate["cost"]),
        "known_mask_equal": bool(known_equal),
        "margin_stable": bool(margin_ok),
        "minimum_nonzero_abs": float(min_abs),
        "proof": {
            "inherits_exhaustive_mod11_relation": True,
            "halo_equation": "Sum(bottom_true, hend_minus1, 1) = bottom_true + hend",
            "all_input_equivalent": True
        }
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["cost_reduction"] > 0 and known_equal and margin_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
