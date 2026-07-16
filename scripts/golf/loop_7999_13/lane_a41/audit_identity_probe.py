#!/usr/bin/env python3
"""Known-corpus dual-ORT differential audit for exact Identity bypass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / "inputs/neurogolf-2026/task366.json"
MODELS = {
    "authority": HERE / "baseline_task366.onnx",
    "identity_bypass": HERE / "probe_identity.onnx",
}


def encode(grid) -> np.ndarray:
    h, w = len(grid), len(grid[0])
    if h > 30 or w > 30:
        raise ValueError("oversize")
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            value[0, int(color), r, c] = 1.0
    return value


def session(path: Path, mode: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    else:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


data = json.loads(DATA.read_text())
examples = [item for split in ("train", "test", "arc-gen") for item in data[split]]
rows = []
for mode in ("disabled", "default"):
    runners = {name: session(path, mode) for name, path in MODELS.items()}
    row = {
        "mode": mode,
        "total_seen": len(examples),
        "executable": 0,
        "skipped_oversize": 0,
        "authority_right": 0,
        "candidate_right": 0,
        "candidate_wrong": 0,
        "authority_errors": 0,
        "candidate_errors": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "first_failure": None,
    }
    for index, example in enumerate(examples):
        try:
            input_value = encode(example["input"])
            expected = encode(example["output"]).astype(bool)
        except ValueError:
            row["skipped_oversize"] += 1
            continue
        row["executable"] += 1
        values = {}
        for name, runner in runners.items():
            try:
                values[name] = np.asarray(runner.run(["output"], {"input": input_value})[0])
                if np.array_equal(values[name] > 0, expected):
                    row[f"{'authority' if name == 'authority' else 'candidate'}_right"] += 1
                elif name == "identity_bypass":
                    row["candidate_wrong"] += 1
            except Exception as exc:  # noqa: BLE001
                row[f"{'authority' if name == 'authority' else 'candidate'}_errors"] += 1
                row["first_failure"] = row["first_failure"] or {"index": index, "error": repr(exc)}
        if len(values) == 2:
            if np.array_equal(values["authority"], values["identity_bypass"]):
                row["raw_equal"] += 1
            if np.array_equal(values["authority"] > 0, values["identity_bypass"] > 0):
                row["threshold_equal"] += 1
            elif row["first_failure"] is None:
                row["first_failure"] = {"index": index, "kind": "differential"}
    row["pass"] = (
        row["candidate_right"] == row["executable"]
        and row["candidate_errors"] == 0
        and row["raw_equal"] == row["executable"]
    )
    rows.append(row)
    print(row)

(HERE / "identity_probe_known_dual.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
