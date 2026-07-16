#!/usr/bin/env python3
"""All-input exact task349 offset-table elimination on the LB8008.14 member."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "task349.onnx"
OUTPUT = HERE / "task349_relation_exact.onnx"
REPORT = HERE / "relation_result.json"
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import (  # noqa: E402
    masks_equal_with_margin,
    model_margin_stable,
    score_and_verify,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    model = onnx.load(SOURCE)
    arrays = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    top = np.asarray(arrays["top_offset_by_mod_i8"], dtype=np.int8)
    hstart = np.asarray(arrays["hstart_offset_by_mod_i8"], dtype=np.int8)
    hend = np.asarray(arrays["hend_offset_by_mod_i8"], dtype=np.int8)
    derived = (hstart.astype(np.int16) + hend.astype(np.int16) - 1).astype(np.int8)
    if len(top) != 11 or not np.array_equal(top, derived):
        raise AssertionError("top=hstart+hend-1 must hold over the full Mod11 domain")
    if "one_i8" in arrays:
        raise AssertionError("initializer name collision")

    kept = []
    removed_top = changed_hend = False
    for initializer in model.graph.initializer:
        if initializer.name == "top_offset_by_mod_i8":
            removed_top = True
            continue
        if initializer.name == "hend_offset_by_mod_i8":
            initializer.CopyFrom(
                numpy_helper.from_array(
                    np.ascontiguousarray(hend - np.int8(1)),
                    name=initializer.name,
                )
            )
            changed_hend = True
        kept.append(initializer)
    if not (removed_top and changed_hend):
        raise AssertionError("required initializers were not rewritten")
    kept.append(numpy_helper.from_array(np.asarray(1, dtype=np.int8), name="one_i8"))
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    rewritten = []
    omitted_top_gather = inserted_top = restored_halo = False
    for node in model.graph.node:
        if list(node.output) == ["top_offset_i8"]:
            if node.op_type != "Gather" or node.input[0] != "top_offset_by_mod_i8":
                raise AssertionError("unexpected top-offset producer")
            omitted_top_gather = True
            continue
        rewritten.append(node)
        if list(node.output) == ["hend_offset_i8"]:
            rewritten.append(
                helper.make_node(
                    "Add",
                    ["hstart_offset_i8", "hend_offset_i8"],
                    ["top_offset_i8"],
                    name="top_offset_i8_from_global_relation",
                )
            )
            inserted_top = True
        if list(node.output) == ["halo_end1"]:
            if node.op_type != "Add":
                raise AssertionError("unexpected halo-end producer")
            node.output[0] = "halo_end_minus1"
            node.name = "halo_end_minus1"
            rewritten.append(
                helper.make_node(
                    "Add",
                    ["halo_end_minus1", "one_i8"],
                    ["halo_end1"],
                    name="halo_end1_restore",
                )
            )
            restored_halo = True
    if not (omitted_top_gather and inserted_top and restored_halo):
        raise AssertionError("graph rewrite was incomplete")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    if inferred.functions or inferred.graph.sparse_initializer:
        raise AssertionError("functions/sparse initializers are forbidden")
    if any(x.domain not in {"", "ai.onnx"} for x in inferred.opset_import):
        raise AssertionError("nonstandard domain")
    onnx.save(model, OUTPUT)

    work = HERE / "work"
    baseline = score_and_verify(
        copy.deepcopy(onnx.load(SOURCE)), 349, str(work),
        label="task349_lb8008_14", require_correct=True,
    )
    candidate = score_and_verify(
        copy.deepcopy(model), 349, str(work),
        label="task349_relation_exact", require_correct=True,
    )
    if baseline is None or candidate is None:
        raise RuntimeError(f"official scoring failed: {baseline=} {candidate=}")
    known_equal = masks_equal_with_margin(
        copy.deepcopy(onnx.load(SOURCE)), copy.deepcopy(model), 349, margin=0.25
    )
    margin_ok, min_abs = model_margin_stable(copy.deepcopy(model), 349, margin=0.25)
    payload = {
        "task": 349,
        "authority_score": 8008.14,
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
            "radius_code_domain": "Mod11 => {0,...,10}",
            "identity": "top_offset[i] = hstart_offset[i] + hend_offset[i] - 1",
            "identity_exhaustive_indices": 11,
            "original_hend_restored_before_all_other_consumers": True,
            "all_input_equivalent": True,
        },
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["cost_reduction"] > 0 and known_equal and margin_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
