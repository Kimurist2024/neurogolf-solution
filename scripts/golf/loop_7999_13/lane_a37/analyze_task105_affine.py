#!/usr/bin/env python3
"""Inspect whether task105's root_2 Add is linear on the reachable v_pair state."""

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
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from lib import scoring  # noqa: E402


TASK = 105
SOURCE = HERE.parent / "lane_initializer_contraction_wave17" / "task105_combined.onnx"
OUTPUT = HERE / "task105_affine_audit.json"


def main() -> None:
    model = onnx.load(SOURCE)
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    for name in ("v_pair", "pair_v", "h_pair", "pair_h", "pair_top", "pair_bottom", "pair_right"):
        traced.graph.output.append(copy.deepcopy(typed[name]))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    runner = ort.InferenceSession(traced.SerializeToString(), options)
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    module = importlib.import_module(f"task_{task_map[f'{TASK:03d}']}")
    random.seed(371_105)
    examples = []
    for subset in ("train", "test", "arc-gen"):
        examples.extend(scoring.load_examples(TASK)[subset])
    examples.extend(module.generate() for _ in range(500))
    names = [value.name for value in traced.graph.output]
    rows: dict[str, list[list[float]]] = {name: [] for name in names}
    for example in examples:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        values = runner.run(names, {"input": benchmark["input"]})
        for name, value in zip(names, values):
            rows[name].append(np.asarray(value).reshape(-1).astype(np.float64).tolist())
    unique = {
        name: sorted({tuple(value) for value in values})
        for name, values in rows.items()
    }
    x = np.asarray(rows["v_pair"], dtype=np.float64)
    y = np.asarray(rows["pair_v"], dtype=np.float64)
    transform, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    rebuilt = x @ transform
    report = {
        "task": TASK,
        "source": str(SOURCE.relative_to(ROOT)),
        "samples": len(x),
        "unique_states": {name: [list(value) for value in values] for name, values in unique.items()},
        "linear_transform_x_times_M": transform.tolist(),
        "linear_transform_exact_on_samples": bool(np.array_equal(rebuilt, y)),
        "linear_transform_max_abs_error": float(np.max(np.abs(rebuilt - y))),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
