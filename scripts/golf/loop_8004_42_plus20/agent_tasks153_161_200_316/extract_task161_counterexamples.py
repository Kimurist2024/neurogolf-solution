#!/usr/bin/env python3
"""Persist the first generator-legal task161 failures of the authority member."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def main() -> int:
    data = (HERE / "baseline/task161.onnx").read_bytes()
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    generator = importlib.import_module("task_6cdd2623")
    common = importlib.import_module("common")
    rows = []
    for seed in (153_161_200, 316_200_161):
        random.seed(seed)
        common.random.seed(seed)
        for index in range(3000):
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
            value = np.asarray(
                session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0]
            )
            expected = benchmark["output"].astype(bool)
            observed = value > 0
            if not np.array_equal(observed, expected):
                grid = example["input"]
                present = sorted({cell for row in grid for cell in row if cell})
                counts = {str(color): sum(row.count(color) for row in grid) for color in present}
                rows.append({
                    "seed": seed,
                    "index": index,
                    "input_height": len(grid),
                    "input_width": len(grid[0]),
                    "nonzero_color_counts": counts,
                    "expected_nonzero_cells": int(np.count_nonzero(expected[:, 1:])),
                    "observed_nonzero_cells": int(np.count_nonzero(observed[:, 1:])),
                    "threshold_differences": int(np.count_nonzero(observed != expected)),
                    "input": grid,
                    "expected_output": example["output"],
                })
                break
    output = HERE / "evidence/task161_counterexamples.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
