#!/usr/bin/env python3
"""Fast one-example rejection gate for the two interrupted task359 probes."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    clean = scoring.sanitize_model(copy.deepcopy(model))
    if clean is None:
        raise RuntimeError("sanitize_model rejected candidate")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(clean.SerializeToString(), options)
    example = scoring.load_examples(359)["train"][0]
    benchmark = scoring.convert_to_numpy(example)
    if benchmark is None:
        raise RuntimeError("failed to convert known example")
    actual = session.run(["output"], {"input": benchmark["input"]})[0] > 0
    expected = benchmark["output"] > 0
    return {
        "candidate": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "checker_full": True,
        "strict_shape_inference_data_prop": True,
        "known_probe": "train[0]",
        "runtime_errors": 0,
        "threshold_equal": bool(np.array_equal(actual, expected)),
        "different_cells": int(np.count_nonzero(actual != expected)),
        "verdict": "REJECT" if not np.array_equal(actual, expected) else "NEEDS_FULL_GATE",
    }


def main() -> None:
    source = HERE.parent / "agent_changed_tasks/broadcast_prunes"
    paths = [
        source / "task359_M_axis1_idx0.onnx",
        source / "task359_M_axis1_idx1.onnx",
    ]
    rows = [run(path) for path in paths]
    (HERE / "task359_remaining_quick_reject.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
