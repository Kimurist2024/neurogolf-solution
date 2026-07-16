#!/usr/bin/env python3
"""Replace task191's dynamic scalar square sizes by explicit [H,W] pairs."""

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
SOURCE = HERE / "task191.onnx"
OUTPUT = HERE / "task191_explicit_square_shapes.onnx"
RESULT = HERE / "identity_shape2_result.json"
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import masks_equal_with_margin, score_and_verify  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    model = onnx.load(SOURCE)
    found_init = False
    for initializer in model.graph.initializer:
        if initializer.name == "shape30hw":
            values = numpy_helper.to_array(initializer)
            if not np.array_equal(values, np.asarray([30], np.int64)):
                raise AssertionError(f"unexpected shape30hw: {values}")
            initializer.CopyFrom(
                numpy_helper.from_array(np.asarray([30, 30], np.int64), "shape30hw")
            )
            found_init = True
    if not found_init:
        raise AssertionError("shape30hw initializer not found")

    rewritten = []
    removed = False
    for node in model.graph.node:
        if node.op_type == "Identity" and list(node.input) == ["shape30hw"] and list(node.output) == ["shape30hw_dyn"]:
            removed = True
            continue
        for i, name in enumerate(node.input):
            if name == "shape30hw_dyn":
                node.input[i] = "shape30hw"
        rewritten.append(node)
    if not removed:
        raise AssertionError("shape Identity not found")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    # Stale value_info belongs to the authority's deliberate shape cloak and
    # is not trusted as proof; preserve it byte-for-byte except for the removed
    # tensor name, then let the official profiler/runtime decide fail-closed.
    kept_vi = [x for x in model.graph.value_info if x.name != "shape30hw_dyn"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    onnx.save(model, OUTPUT)

    errors: list[str] = []
    checker = strict = True
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    except Exception as exc:
        checker = False
        errors.append(f"checker: {exc!r}")
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    except Exception as exc:
        strict = False
        errors.append(f"strict: {exc!r}")

    baseline = score_and_verify(
        copy.deepcopy(onnx.load(SOURCE)), 191, str(HERE / "work_shape2"),
        label="task191_lb8008_14", require_correct=True,
    )
    candidate = None
    try:
        candidate = score_and_verify(
            copy.deepcopy(model), 191, str(HERE / "work_shape2"),
            label="task191_explicit_square_shapes", require_correct=True,
        )
    except BaseException as exc:
        errors.append(f"score: {exc!r}")
    known_equal = False
    if candidate is not None:
        try:
            known_equal = bool(masks_equal_with_margin(
                copy.deepcopy(onnx.load(SOURCE)), copy.deepcopy(model), 191, margin=0.25
            ))
        except BaseException as exc:
            errors.append(f"known_equal: {exc!r}")
    gain = None
    if baseline and candidate:
        gain = math.log(baseline["cost"] / candidate["cost"])
    payload = {
        "source_sha256": digest(SOURCE),
        "candidate_sha256": digest(OUTPUT),
        "checker_full": checker,
        "strict_data_prop": strict,
        "baseline": baseline,
        "candidate": candidate,
        "known_mask_equal": known_equal,
        "projected_gain": gain,
        "errors": errors,
        "decision": (
            "PROBE" if candidate and baseline and candidate["cost"] < baseline["cost"] and known_equal
            else "REJECT"
        )
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["decision"] == "PROBE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
