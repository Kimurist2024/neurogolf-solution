#!/usr/bin/env python3
"""Crash-isolated runtime/cost probe for an intentionally questionable model."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.lib import scoring  # noqa: E402


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.model)
    examples = scoring.load_examples(args.task)
    rows = []
    for disabled, label in ((True, "disable_all"), (False, "default")):
        runner = session(model, disabled)
        right = wrong = errors = 0
        for subset in ("train", "test", "arc-gen"):
            for example in examples[subset]:
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    continue
                try:
                    raw = runner.run(["output"], {"input": benchmark["input"]})[0]
                    right += int(np.array_equal(raw > 0.0, benchmark["output"] > 0.0))
                    wrong += int(not np.array_equal(raw > 0.0, benchmark["output"] > 0.0))
                except Exception:  # noqa: BLE001
                    errors += 1
        rows.append({"mode": label, "right": right, "wrong": wrong, "errors": errors})
    memory, parameters, cost = map(int, cost_of(str(args.model)))
    print(json.dumps({"known_dual": rows, "memory": memory, "parameters": parameters, "cost": cost}))


if __name__ == "__main__":
    main()
