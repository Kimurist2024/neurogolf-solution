#!/usr/bin/env python3
"""Independent four-config fresh audit of the immutable task222 authority."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


MODEL_PATH = HERE / "authority" / "task222.onnx"
GENERATOR_PATH = ROOT / "inputs/arc-gen-repo/tasks/task_91714a58.py"
COUNT = 1000
SEED = 222_221_001
CONFIGS = (
    ("disable_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
    ("disable_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
    ("default_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
    ("default_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
)


def load_generator():
    sys.path.insert(0, str(GENERATOR_PATH.parent.parent))
    spec = importlib.util.spec_from_file_location("task222_fresh_generator", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_session(model: onnx.ModelProto, level: Any, threads: int):
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def main() -> None:
    generator = load_generator()
    random.seed(SEED)
    cases = []
    for _ in range(COUNT):
        converted = scoring.convert_to_numpy(generator.generate())
        assert converted is not None
        cases.append(converted)
    model = onnx.load(MODEL_PATH)
    raw_by_config = []
    report = {
        "task": 222,
        "authority_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "generator_source": str(GENERATOR_PATH.relative_to(ROOT)),
        "seed": SEED,
        "count": COUNT,
        "configs": [],
    }
    for label, level, threads in CONFIGS:
        session = make_session(model, level, threads)
        right = wrong = errors = nonfinite = near_positive = 0
        first_wrong = None
        raws = []
        for index, case in enumerate(cases):
            try:
                raw = session.run(["output"], {"input": case["input"]})[0]
            except Exception as exc:
                errors += 1
                if first_wrong is None:
                    first_wrong = {
                        "index": index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                continue
            raws.append(raw)
            nonfinite += int(np.count_nonzero(~np.isfinite(raw)))
            near_positive += int(np.count_nonzero((raw > 0.0) & (raw < 0.25)))
            if np.array_equal(raw > 0.0, case["output"]):
                right += 1
            else:
                wrong += 1
                if first_wrong is None:
                    diff = np.argwhere((raw > 0.0) != case["output"])
                    first_wrong = {
                        "index": index,
                        "different_cells": int(diff.shape[0]),
                        "first_index": diff[0].tolist() if diff.size else None,
                    }
        cross_raw_equal = None
        if len(raws) == COUNT:
            if not raw_by_config:
                cross_raw_equal = True
            else:
                cross_raw_equal = all(
                    np.array_equal(left, right_raw)
                    for left, right_raw in zip(raw_by_config[0], raws, strict=True)
                )
            raw_by_config.append(raws)
        report["configs"].append(
            {
                "config": label,
                "right": right,
                "wrong": wrong,
                "errors": errors,
                "nonfinite": nonfinite,
                "near_positive": near_positive,
                "cross_config_raw_equal_to_disable_t1": cross_raw_equal,
                "first_wrong": first_wrong,
            }
        )
        print(label, right, wrong, errors, flush=True)
    (HERE / "evidence" / "authority_fresh_four_config.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
