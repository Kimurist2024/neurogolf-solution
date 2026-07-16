#!/usr/bin/env python3
"""Targeted task354 reductions rebased on the verified 8012.15 authority.

The authority implementation already has a compact row/column routing engine.
This lane tests two spec-preserving reductions independently and together:

* route reuse: the colour selected for column 4 (or columns 7--9) is also the
  right answer at the neighbouring boundary whenever that boundary's anchor
  pixel is gray.  This removes two redundant boolean ``And`` tensors.
* transpose-before-mask: transpose the 10-element top row before broadcasting
  it over the three bands.  This replaces a 30-element transpose output with a
  10-element transpose output.

The script only writes lane-local candidates/evidence.  It never mutates the
root submission or a checkpoint under ``others/``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "others/71502/task354_improved(4).onnx"
OUT = HERE / "candidates"
WORK = HERE / "work"

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def node_by_output(model: onnx.ModelProto, output: str) -> onnx.NodeProto:
    matches = [node for node in model.graph.node if output in node.output]
    if len(matches) != 1:
        raise RuntimeError(f"expected one producer for {output}, got {len(matches)}")
    return matches[0]


def remove_node(model: onnx.ModelProto, output: str) -> None:
    node = node_by_output(model, output)
    model.graph.node.remove(node)


def route_reuse(model: onnx.ModelProto) -> None:
    """Remove boundary conjunctions using already-routed neighbours.

    At column 3, a gray anchor at column 4 belongs to either the middle box or
    (when the middle box is inactive) the left box.  ``u4`` already expresses
    that exact priority.  The symmetric statement holds for column 6 using
    the column-7 anchor and ``u79``.  Blank cells are irrelevant because only
    the gray mask is scattered into the final output.
    """

    h3 = node_by_output(model, "h3")
    h6 = node_by_output(model, "h6")
    if h3.op_type != "Where" or h6.op_type != "Where":
        raise RuntimeError("unexpected boundary selector")
    h3.input[:] = ["g4b", "u4", "rc0"]
    h6.input[:] = ["g7b", "u79", "rc1"]
    remove_node(model, "mid_at3")
    remove_node(model, "right_at6")

    # The original graph computes the conjunction immediately before its
    # boundary selector, while u4/u79 originally appear immediately after it.
    # Move the rewritten selectors behind their newly reused inputs.
    remaining = [node for node in model.graph.node if node not in (h3, h6)]
    reordered: list[onnx.NodeProto] = []
    for node in remaining:
        reordered.append(node)
        if "u4" in node.output:
            reordered.append(h3)
        if "u79" in node.output:
            reordered.append(h6)
    del model.graph.node[:]
    model.graph.node.extend(reordered)


def transpose_before_mask(model: onnx.ModelProto) -> None:
    """Broadcast a 10-element transpose instead of transposing 30 elements."""

    band = next(init for init in model.graph.initializer if init.name == "band")
    band_array = numpy_helper.to_array(band)
    if band_array.shape != (1, 1, 3, 10):
        raise RuntimeError(f"unexpected band shape {band_array.shape}")
    band.CopyFrom(
        numpy_helper.from_array(
            np.transpose(band_array, (0, 1, 3, 2)).copy(), name="band_t"
        )
    )

    mask = node_by_output(model, "band_code")
    transpose = node_by_output(model, "band_code_t")
    if mask.op_type != "Where" or transpose.op_type != "Transpose":
        raise RuntimeError("unexpected band routing nodes")

    # Re-purpose the two existing nodes to preserve topological order.
    mask.op_type = "Transpose"
    mask.input[:] = ["top"]
    mask.output[:] = ["top_t"]
    del mask.attribute[:]
    mask.attribute.extend([helper.make_attribute("perm", [0, 1, 3, 2])])

    transpose.op_type = "Where"
    transpose.input[:] = ["band_t", "top_t", "z0"]
    transpose.output[:] = ["band_code_t"]
    del transpose.attribute[:]


def build(label: str) -> onnx.ModelProto:
    model = onnx.load(BASE)
    if label in {"route_reuse", "combined"}:
        route_reuse(model)
    if label in {"transpose_before_mask", "combined"}:
        transpose_before_mask(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def main() -> int:
    onnxruntime.set_default_logger_severity(4)
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    base_model = onnx.load(BASE)
    base_result = scoring.score_and_verify(
        copy.deepcopy(base_model), 354, str(WORK), "base", require_correct=True
    )
    if base_result is None:
        raise RuntimeError("authority failed local scorer")

    rows = []
    for label in ("route_reuse", "transpose_before_mask", "combined"):
        model = build(label)
        data = model.SerializeToString()
        path = OUT / f"task354_{label}.onnx"
        path.write_bytes(data)
        result = scoring.score_and_verify(
            copy.deepcopy(model), 354, str(WORK), label, require_correct=True
        )
        rows.append(
            {
                "label": label,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(data),
                "serialized_bytes": len(data),
                "known_exact": result is not None and bool(result.get("correct")),
                "profile": result,
                "strict_lower": result is not None
                and int(result["cost"]) < int(base_result["cost"]),
                "cost_reduction": None
                if result is None
                else int(base_result["cost"]) - int(result["cost"]),
            }
        )

    evidence = {
        "task": 354,
        "authority": str(BASE.relative_to(ROOT)),
        "authority_sha256": sha256(BASE.read_bytes()),
        "authority_profile": base_result,
        "rows": rows,
        "protected_writes": "none; lane-local candidates only",
    }
    (HERE / "search_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
