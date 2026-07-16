#!/usr/bin/env python3
"""Constant-fold task138's Shape(qcol) into an exact dense initializer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import (  # noqa: E402
    masks_equal_with_margin,
    model_margin_stable,
    score_and_verify,
)


BASE = ROOT / "scripts/golf/loop_7999_13/lane_rebuild_b2/baseline_task138.onnx"
OUTPUT = HERE / "candidates/task138_fold_shape_qcol.onnx"
REPORT = HERE / "task138_shape_fold_report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    base = onnx.load(BASE)
    model = copy.deepcopy(base)
    shape_nodes = [n for n in model.graph.node if n.output and n.output[0] == "q_abs_shape"]
    if len(shape_nodes) != 1 or shape_nodes[0].op_type != "Shape" or list(shape_nodes[0].input) != ["qcol"]:
        raise AssertionError("unexpected q_abs_shape producer")

    qcol = next(x for x in model.graph.initializer if x.name == "qcol")
    qcol_shape = list(qcol.dims)
    if qcol_shape != [1, 1, 1, 30]:
        raise AssertionError(f"unexpected qcol shape: {qcol_shape}")
    # All three CenterCropPad consumers specify exactly one axis.  The
    # incumbent passes Shape(qcol)=[1,1,1,30] while declaring its type [1]; ORT
    # consumes only the first value.  Feed the schema-valid one-element target
    # [1] directly, preserving that effective value without the mismatch.
    target_extent = np.asarray([qcol_shape[0]], dtype=np.int64)

    kept_nodes = [n for n in model.graph.node if n is not shape_nodes[0]]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    # The dense initializer honestly has the one value required by the one
    # explicit axis on each CenterCropPad consumer.
    kept_vi = [v for v in model.graph.value_info if v.name != "q_abs_shape"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    model.graph.initializer.append(numpy_helper.from_array(target_extent, name="q_abs_shape"))

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    if inferred.functions or inferred.graph.sparse_initializer:
        raise AssertionError("functions/sparse initializers forbidden")
    if any(op.domain not in {"", "ai.onnx"} for op in inferred.opset_import):
        raise AssertionError("foreign domain")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)

    work = HERE / "work"
    base_score = score_and_verify(copy.deepcopy(base), 138, str(work), label="base138_fold", require_correct=True)
    score = score_and_verify(copy.deepcopy(model), 138, str(work), label="shape138_fold", require_correct=True)
    if base_score is None or score is None:
        raise RuntimeError(f"score failed: {base_score=} {score=}")
    margin_ok, min_abs = model_margin_stable(copy.deepcopy(model), 138, margin=0.25)
    report = {
        "task": 138,
        "rewrite": "Shape(qcol) -> schema-valid dense int64 one-axis target [1]",
        "baseline": {**base_score, "sha256": sha256(BASE)},
        "candidate": {
            **score,
            "path": str(OUTPUT.relative_to(ROOT)),
            "sha256": sha256(OUTPUT),
            "bytes": OUTPUT.stat().st_size,
        },
        "delta_cost": score["cost"] - base_score["cost"],
        "projected_score_gain": math.log(base_score["cost"] / score["cost"]),
        "known_mask_equal_with_margin": masks_equal_with_margin(
            copy.deepcopy(base), copy.deepcopy(model), 138, margin=0.25
        ),
        "margin": {"stable": margin_ok, "min_nonzero_abs": min_abs},
        "structure": {
            "checker_full": True,
            "strict_shape_inference": True,
            "domains": sorted({op.domain for op in inferred.opset_import}),
            "functions": len(inferred.functions),
            "sparse_initializers": len(inferred.graph.sparse_initializer),
            "inputs": len(inferred.graph.input),
            "outputs": len(inferred.graph.output),
            "q_abs_shape_initializer_dims": list(next(x for x in inferred.graph.initializer if x.name == "q_abs_shape").dims),
            "q_abs_shape_value_info_count": sum(v.name == "q_abs_shape" for v in inferred.graph.value_info),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if score["cost"] < base_score["cost"] and report["known_mask_equal_with_margin"] and margin_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
