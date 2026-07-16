#!/usr/bin/env python3
"""Dual-ORT known and raw-authority differential audit for task366 controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODELS = {
    "authority": HERE / "baseline_task366.onnx",
    "identity_bypass": HERE / "probe_identity.onnx",
    "truthful_annotations": HERE / "truthful_annotation_control.onnx",
    "truthful_identity_bypass": HERE / "truthful_identity_bypass.onnx",
    "historical_7985": ROOT / "others/2/1203/task366_improved.onnx",
    "historical_7916": ROOT / "others/2/1203/task366_further_improved.onnx",
    "historical_7646": ROOT / "others/2/1201/7120/task366_further_improved.onnx",
}


def encode(grid) -> np.ndarray:
    if len(grid) > 30 or len(grid[0]) > 30:
        raise ValueError("oversize")
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            value[0, int(color), r, c] = 1.0
    return value


def make_session(path: Path, mode: str):
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "disabled"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


data = json.loads((ROOT / "inputs/neurogolf-2026/task366.json").read_text())
examples = [example for split in ("train", "test", "arc-gen") for example in data[split]]
results = []
for mode in ("disabled", "default"):
    sessions = {}
    session_errors = {}
    for label, path in MODELS.items():
        try:
            sessions[label] = make_session(path, mode)
        except Exception as exc:  # noqa: BLE001
            session_errors[label] = f"{type(exc).__name__}: {exc}"
    rows = {
        label: {
            "label": label,
            "mode": mode,
            "known_right": 0,
            "known_wrong": 0,
            "runtime_errors": 0,
            "raw_equal_authority": 0,
            "threshold_equal_authority": 0,
            "first_failure": None,
            "session_error": session_errors.get(label),
        }
        for label in MODELS
    }
    executable = 0
    skipped = 0
    for index, example in enumerate(examples):
        try:
            input_value = encode(example["input"])
            expected = encode(example["output"]).astype(bool)
        except ValueError:
            skipped += 1
            continue
        executable += 1
        outputs = {}
        for label, runner in sessions.items():
            try:
                raw = np.asarray(runner.run(["output"], {"input": input_value})[0])
                outputs[label] = raw
                if np.array_equal(raw > 0, expected):
                    rows[label]["known_right"] += 1
                else:
                    rows[label]["known_wrong"] += 1
                    rows[label]["first_failure"] = rows[label]["first_failure"] or {"index": index, "kind": "known"}
            except Exception as exc:  # noqa: BLE001
                rows[label]["runtime_errors"] += 1
                rows[label]["first_failure"] = rows[label]["first_failure"] or {"index": index, "error": f"{type(exc).__name__}: {exc}"}
        base = outputs.get("authority")
        if base is not None:
            for label, raw in outputs.items():
                if np.array_equal(raw, base):
                    rows[label]["raw_equal_authority"] += 1
                if np.array_equal(raw > 0, base > 0):
                    rows[label]["threshold_equal_authority"] += 1
    for row in rows.values():
        row["executable"] = executable
        row["skipped_oversize"] = skipped
        row["pass_known"] = row["known_right"] == executable and row["known_wrong"] == 0 and row["runtime_errors"] == 0
        row["pass_raw_equivalence"] = row["raw_equal_authority"] == executable
        results.append(row)
        print(row)

(HERE / "known_dual_all.json").write_text(json.dumps({"rows": results}, indent=2) + "\n")
