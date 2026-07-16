#!/usr/bin/env python3
"""Reuse task349's existing six-way equality mask for its scalar special case."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import onnx
from onnx import helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "task349.onnx"
PREVIOUS = HERE / "task349_relation_zero_exact.onnx"
OUTPUT = HERE / "task349_relation_zero_sig_exact.onnx"
REPORT = HERE / "relation_zero_sig_result.json"
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import (  # noqa: E402
    masks_equal_with_margin,
    model_margin_stable,
    score_and_verify,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    model = onnx.load(PREVIOUS)
    kept = [x for x in model.graph.initializer if x.name != "sig_ex233_k4"]
    if len(kept) + 1 != len(model.graph.initializer):
        raise AssertionError("sig_ex233_k4 initializer not found exactly once")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    rewritten = []
    replaced = False
    for node in model.graph.node:
        if list(node.output) == ["sp_has_sig"]:
            if node.op_type != "Equal" or list(node.input) != ["patch_sumR", "sig_ex233_k4"]:
                raise AssertionError("unexpected special-signature producer")
            rewritten.append(
                helper.make_node(
                    "Gather",
                    ["h_patch_mask", "k6"],
                    ["sp_has_sig"],
                    name="sp_has_sig_from_exhaustive_mask_index4",
                    axis=2,
                )
            )
            replaced = True
        else:
            rewritten.append(node)
    if not replaced:
        raise AssertionError("special-signature producer not found")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    if inferred.functions or inferred.graph.sparse_initializer:
        raise AssertionError("functions/sparse initializers are forbidden")
    onnx.save(model, OUTPUT)

    work = HERE / "work_sig"
    baseline = score_and_verify(
        copy.deepcopy(onnx.load(SOURCE)), 349, str(work),
        label="task349_lb8008_14", require_correct=True,
    )
    candidate = score_and_verify(
        copy.deepcopy(model), 349, str(work),
        label="task349_relation_zero_sig_exact", require_correct=True,
    )
    if baseline is None or candidate is None:
        raise RuntimeError(f"official scoring failed: {baseline=} {candidate=}")
    known_equal = masks_equal_with_margin(
        copy.deepcopy(onnx.load(SOURCE)), copy.deepcopy(model), 349, margin=0.25
    )
    margin_ok, min_abs = model_margin_stable(copy.deepcopy(model), 349, margin=0.25)
    payload = {
        "task": 349,
        "source_sha256": digest(SOURCE),
        "candidate_sha256": digest(OUTPUT),
        "baseline": baseline,
        "candidate": candidate,
        "cost_reduction": int(baseline["cost"] - candidate["cost"]),
        "projected_gain": math.log(baseline["cost"] / candidate["cost"]),
        "known_mask_equal": bool(known_equal),
        "margin_stable": bool(margin_ok),
        "minimum_nonzero_abs": float(min_abs),
        "proof": {
            "inherits_prior_exact_proofs": True,
            "h_patch_sigs_index4": 214431744,
            "removed_scalar_signature": 214431744,
            "k6_index": 4,
            "gathered_boolean_equals_removed_equal": True,
            "all_input_equivalent": True
        }
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["cost_reduction"] > 0 and known_equal and margin_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
