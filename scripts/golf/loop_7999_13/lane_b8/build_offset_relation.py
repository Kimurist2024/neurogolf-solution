#!/usr/bin/env python3
"""Remove task349's redundant top-offset table using an exact relation.

For every reachable radius, ``top = hstart + (hend - 1)``.  Store hend-1,
derive top with the same int8 Add already used by the incumbent, and restore
hend with one final Add.  This removes nine table parameters at the cost of
one scalar parameter and one six-byte intermediate: net cost -2.
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
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import (  # noqa: E402
    masks_equal_with_margin,
    model_margin_stable,
    score_and_verify,
)


SOURCE = HERE / "candidates" / "task349_radius_tables_len9.onnx"
OUTPUT = HERE / "candidates" / "task349_radius_tables_len9_top_relation.onnx"
BASE = HERE / "baseline_task349.onnx"
REPORT = HERE / "offset_relation_report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    model = onnx.load(SOURCE)
    expected_hend = np.asarray([6, 1, 5, 2, 3, 1, 1, 1, 4], dtype=np.int8)

    kept = []
    saw_top = saw_hend = False
    for initializer in model.graph.initializer:
        if initializer.name == "top_offset_by_mod_i8":
            saw_top = True
            continue
        if initializer.name == "hend_offset_by_mod_i8":
            actual = numpy_helper.to_array(initializer)
            if not np.array_equal(actual, expected_hend):
                raise AssertionError(f"unexpected hend table: {actual}")
            initializer.CopyFrom(
                numpy_helper.from_array(np.ascontiguousarray(actual - np.int8(1)), name=initializer.name)
            )
            saw_hend = True
        kept.append(initializer)
    if not (saw_top and saw_hend):
        raise AssertionError("required offset initializers not found")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(1, dtype=np.int8), name="one_i8"))

    rewritten = []
    pending_top = False
    for node in model.graph.node:
        if node.output and node.output[0] == "top_offset_i8":
            if node.op_type != "Gather":
                raise AssertionError("unexpected top producer")
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
                    name="top_offset_i8",
                )
            )
            pending_top = False
        if node.output and node.output[0] == "halo_end1":
            if node.op_type != "Add":
                raise AssertionError("unexpected halo_end producer")
            node.output[0] = "halo_end_minus1"
            node.name = "halo_end_minus1"
            rewritten.append(
                helper.make_node(
                    "Add",
                    ["halo_end_minus1", "one_i8"],
                    ["halo_end1"],
                    name="halo_end1",
                )
            )
    if pending_top:
        raise AssertionError("failed to insert derived top")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if inferred.functions or inferred.graph.sparse_initializer:
        raise AssertionError("functions/sparse initializer forbidden")
    if any(op.domain not in {"", "ai.onnx"} for op in inferred.opset_import):
        raise AssertionError("foreign opset domain")
    onnx.save(model, OUTPUT)

    base = onnx.load(BASE)
    source = onnx.load(SOURCE)
    work = HERE / "work"
    base_score = score_and_verify(copy.deepcopy(base), 349, str(work), label="base349_rel", require_correct=True)
    source_score = score_and_verify(copy.deepcopy(source), 349, str(work), label="crop349_rel", require_correct=True)
    score = score_and_verify(copy.deepcopy(model), 349, str(work), label="relation349", require_correct=True)
    if base_score is None or source_score is None or score is None:
        raise RuntimeError(f"scoring failed: {base_score=} {source_score=} {score=}")
    margin_ok, min_abs = model_margin_stable(copy.deepcopy(model), 349, margin=0.25)
    report = {
        "task": 349,
        "proof": {
            "reachable_radius_codes": [0, 2, 3, 4, 8],
            "identity": "top_offset = hstart_offset + (hend_offset - 1)",
            "stored_hend_semantics": "hend_offset - 1",
            "new_arithmetic_dtype": "int8",
            "dtype_kernel_precedent": "incumbent already executes int8 Add under ORT_DISABLE_ALL",
        },
        "baseline": {**base_score, "sha256": sha256(BASE)},
        "table_crop": {**source_score, "sha256": sha256(SOURCE)},
        "candidate": {
            **score,
            "path": str(OUTPUT.relative_to(ROOT)),
            "sha256": sha256(OUTPUT),
            "bytes": OUTPUT.stat().st_size,
        },
        "delta_vs_baseline": score["cost"] - base_score["cost"],
        "delta_vs_table_crop": score["cost"] - source_score["cost"],
        "projected_score_gain_vs_baseline": math.log(base_score["cost"] / score["cost"]),
        "known_mask_equal_with_margin": masks_equal_with_margin(
            copy.deepcopy(base), copy.deepcopy(model), 349, margin=0.25
        ),
        "margin": {"stable": margin_ok, "min_nonzero_abs": min_abs},
        "structure": {
            "checker_full": True,
            "strict_shape_inference": True,
            "functions": len(inferred.functions),
            "sparse_initializers": len(inferred.graph.sparse_initializer),
            "domains": sorted({op.domain for op in inferred.opset_import}),
            "inputs": len(inferred.graph.input),
            "outputs": len(inferred.graph.output),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    ok = (
        score["cost"] < source_score["cost"] < base_score["cost"]
        and report["known_mask_equal_with_margin"]
        and margin_ok
        and not inferred.functions
        and not inferred.graph.sparse_initializer
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
