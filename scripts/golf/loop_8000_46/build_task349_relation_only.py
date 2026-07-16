#!/usr/bin/env python3
"""Remove task349's redundant top-offset table without shrinking its domain.

Unlike the earlier table-crop experiment, this rewrite keeps all eleven
indices executable.  The identity ``top = hstart + hend - 1`` holds for every
stored index, so the resulting graph is intended to be raw-equivalent to the
incumbent even on off-generator inputs.
"""

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
ROOT = HERE.parents[2]
SOURCE = HERE / "latest_8002_63_models" / "task349.onnx"
LANE = HERE / "lane_task349_relation_only"
OUTPUT = LANE / "task349_relation_only.onnx"
REPORT = LANE / "build_audit.json"
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import (  # noqa: E402
    masks_equal_with_margin,
    model_margin_stable,
    score_and_verify,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    LANE.mkdir(parents=True, exist_ok=True)
    model = onnx.load(SOURCE)

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    top = np.asarray(initializers["top_offset_by_mod_i8"], dtype=np.int8)
    hstart = np.asarray(initializers["hstart_offset_by_mod_i8"], dtype=np.int8)
    hend = np.asarray(initializers["hend_offset_by_mod_i8"], dtype=np.int8)
    derived = (hstart.astype(np.int16) + hend.astype(np.int16) - 1).astype(np.int8)
    if not np.array_equal(top, derived):
        raise AssertionError(f"offset identity is not global: {top=} {derived=}")
    if not (len(top) == len(hstart) == len(hend) == 11):
        raise AssertionError("the complete eleven-index domain must be retained")

    kept = []
    saw_top = saw_hend = False
    for initializer in model.graph.initializer:
        if initializer.name == "top_offset_by_mod_i8":
            saw_top = True
            continue
        if initializer.name == "hend_offset_by_mod_i8":
            initializer.CopyFrom(
                numpy_helper.from_array(
                    np.ascontiguousarray(hend - np.int8(1)),
                    name=initializer.name,
                )
            )
            saw_hend = True
        kept.append(initializer)
    if not (saw_top and saw_hend):
        raise AssertionError("required offset initializers were not found")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray(1, dtype=np.int8), name="one_i8")
    )

    rewritten = []
    pending_top = False
    restored_halo_end = False
    for node in model.graph.node:
        if node.output and node.output[0] == "top_offset_i8":
            if node.op_type != "Gather":
                raise AssertionError("unexpected top-offset producer")
            pending_top = True
            continue
        rewritten.append(node)
        if node.output and node.output[0] == "hend_offset_i8":
            if not pending_top:
                raise AssertionError("top producer ordering changed")
            rewritten.append(
                helper.make_node(
                    "Add",
                    ["hstart_offset_i8", "hend_offset_i8"],
                    ["top_offset_i8"],
                    name="top_offset_i8_from_relation",
                )
            )
            pending_top = False
        if node.output and node.output[0] == "halo_end1":
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
            restored_halo_end = True
    if pending_top or not restored_halo_end:
        raise AssertionError("relation rewrite was incomplete")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if inferred.functions or inferred.graph.sparse_initializer:
        raise AssertionError("functions and sparse initializers are forbidden")
    domains = sorted({item.domain for item in inferred.opset_import})
    if any(domain not in {"", "ai.onnx"} for domain in domains):
        raise AssertionError(f"foreign domains: {domains}")
    onnx.save(model, OUTPUT)

    work = LANE / "work"
    base_score = score_and_verify(
        copy.deepcopy(onnx.load(SOURCE)),
        349,
        str(work),
        label="base349_relation_only",
        require_correct=True,
    )
    candidate_score = score_and_verify(
        copy.deepcopy(model),
        349,
        str(work),
        label="candidate349_relation_only",
        require_correct=True,
    )
    if base_score is None or candidate_score is None:
        raise RuntimeError(f"scoring failed: {base_score=} {candidate_score=}")
    margin_ok, min_abs = model_margin_stable(copy.deepcopy(model), 349, margin=0.25)
    known_equal = masks_equal_with_margin(
        copy.deepcopy(onnx.load(SOURCE)), copy.deepcopy(model), 349, margin=0.25
    )
    cost_reduction = int(base_score["cost"] - candidate_score["cost"])
    payload = {
        "task": 349,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE),
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "candidate_sha256": digest(OUTPUT),
        "identity_global_all_11_indices": True,
        "table_length_retained": 11,
        "baseline": base_score,
        "candidate_score": candidate_score,
        "cost_reduction": cost_reduction,
        "projected_gain": math.log(base_score["cost"] / candidate_score["cost"]),
        "known_mask_equal_with_margin": known_equal,
        "margin": {"stable": margin_ok, "min_nonzero_abs": min_abs},
        "structure": {
            "checker_full": True,
            "strict_shape_inference": True,
            "functions": len(inferred.functions),
            "sparse_initializers": len(inferred.graph.sparse_initializer),
            "domains": domains,
        },
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if cost_reduction > 0 and known_equal and margin_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
