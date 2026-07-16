#!/usr/bin/env python3
"""Test the apparent task191 shape30 Identity elimination fail-closed."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "task191.onnx"
OUTPUT = HERE / "task191_no_shape_identity.onnx"
RESULT = HERE / "identity_result.json"
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import score_and_verify  # noqa: E402


def main() -> int:
    model = onnx.load(SOURCE)
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
        raise AssertionError("expected Identity not found")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    checker = strict = True
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        checker = False
        errors.append(f"checker: {exc!r}")
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    except Exception as exc:
        strict = False
        errors.append(f"strict: {exc!r}")
    onnx.save(model, OUTPUT)
    baseline = score_and_verify(
        copy.deepcopy(onnx.load(SOURCE)), 191, str(HERE / "work"),
        label="task191_lb8008_14", require_correct=True,
    )
    candidate = None
    try:
        candidate = score_and_verify(
            copy.deepcopy(model), 191, str(HERE / "work"),
            label="task191_no_shape_identity", require_correct=True,
        )
    except BaseException as exc:  # subprocess/runtime evidence, including SystemExit
        errors.append(f"score: {exc!r}")
    payload = {
        "checker_full": checker,
        "strict_data_prop": strict,
        "baseline": baseline,
        "candidate": candidate,
        "errors": errors,
        "decision": "ACCEPT" if candidate and baseline and candidate["cost"] < baseline["cost"] else "REJECT"
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["decision"] == "ACCEPT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
