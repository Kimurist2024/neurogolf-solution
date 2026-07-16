#!/usr/bin/env python3
"""Dual-ORT known-set and official-cost screen for A32 models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known(path: Path, disabled: bool) -> dict[str, object]:
    model = onnx.load(path)
    try:
        session = make_session(model, disabled)
    except Exception as exc:  # noqa: BLE001
        return {"right": 0, "wrong": 0, "errors": 1, "session_error": repr(exc)}
    right = wrong = errors = 0
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(335)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                raw = session.run(["output"], {"input": benchmark["input"]})[0]
                if np.array_equal(raw > 0, benchmark["output"] > 0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001
                errors += 1
    return {"right": right, "wrong": wrong, "errors": errors}


def audit(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    disabled = known(path, True)
    default = known(path, False)
    row: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "params_static": sum(math.prod(item.dims) for item in model.graph.initializer),
        "node_count": len(model.graph.node),
        "einsum_operands": len(model.graph.node[0].input),
        "disable_all": disabled,
        "default": default,
    }
    perfect = all(item == {"right": 266, "wrong": 0, "errors": 0} for item in (disabled, default))
    row["known_perfect"] = perfect
    if perfect:
        row["score"] = scoring.score_and_verify(
            model, 335, str(HERE / "profile"), label=path.stem, require_correct=True
        )
    return row


def main() -> None:
    rows = [audit(path) for path in sorted(HERE.glob("task335_*.onnx"))]
    (HERE / "candidate_screen.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
