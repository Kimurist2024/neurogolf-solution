#!/usr/bin/env python3
"""Two-seed, two-ORT-mode fresh verification for the truthful task237 base."""

from __future__ import annotations

import copy
import hashlib
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


MODEL_PATH = HERE / "baseline/task237.onnx"
OUTPUT_PATH = HERE / "task237_fresh_2seed.json"
SEEDS = (8_004_237, 8_104_237)
COUNT = 5000


def session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )


def main() -> None:
    ort.set_default_logger_severity(4)
    model = onnx.load(MODEL_PATH, load_external_data=False)
    sessions = {
        "disable_all": session(model, True),
        "default": session(model, False),
    }
    generator = importlib.import_module("task_99fa7670")
    payload: dict[str, object] = {
        "task": 237,
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "requested_per_seed": COUNT,
        "seeds": list(SEEDS),
        "runs": [],
    }
    for seed in SEEDS:
        random.seed(seed)
        rows = {
            mode: {
                "right": 0,
                "wrong": 0,
                "errors": 0,
                "shape_errors": 0,
                "small_positive_cells": 0,
                "min_positive": None,
                "max_off_value": None,
                "first_failure": None,
            }
            for mode in sessions
        }
        for index in range(COUNT):
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
            assert benchmark is not None
            expected = benchmark["output"] > 0
            for mode, active_session in sessions.items():
                row = rows[mode]
                try:
                    actual = np.asarray(
                        active_session.run(
                            [active_session.get_outputs()[0].name],
                            {active_session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                    )
                    if actual.shape != (1, 10, 30, 30):
                        row["shape_errors"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "index": index,
                                "kind": "output_shape",
                                "actual": list(actual.shape),
                            }
                        continue
                    threshold = actual > 0
                    if np.array_equal(threshold, expected):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "index": index,
                                "kind": "threshold_mismatch",
                                "different_cells": int(np.count_nonzero(threshold != expected)),
                            }
                    positive = actual[threshold]
                    off = actual[~threshold]
                    if positive.size:
                        minimum = float(positive.min())
                        row["min_positive"] = (
                            minimum
                            if row["min_positive"] is None
                            else min(row["min_positive"], minimum)
                        )
                        row["small_positive_cells"] += int(
                            np.count_nonzero((positive > 0) & (positive < 0.25))
                        )
                    if off.size:
                        maximum = float(off.max())
                        row["max_off_value"] = (
                            maximum
                            if row["max_off_value"] is None
                            else max(row["max_off_value"], maximum)
                        )
                except Exception as exc:  # noqa: BLE001
                    row["errors"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {
                            "index": index,
                            "kind": "runtime_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
            if (index + 1) % 500 == 0:
                print(f"seed={seed} progress={index + 1}/{COUNT}", flush=True)
        for row in rows.values():
            row["perfect"] = (
                row["right"] == COUNT
                and row["wrong"] == 0
                and row["errors"] == 0
                and row["shape_errors"] == 0
                and row["small_positive_cells"] == 0
            )
        payload["runs"].append({"seed": seed, "modes": rows})
        OUTPUT_PATH.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    payload["perfect"] = all(
        mode["perfect"]
        for run in payload["runs"]
        for mode in run["modes"].values()
    )
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
