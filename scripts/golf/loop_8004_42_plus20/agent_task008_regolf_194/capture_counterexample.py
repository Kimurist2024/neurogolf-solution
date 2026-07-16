#!/usr/bin/env python3
"""Capture an explicit generator-reachable failure of the task008 authority."""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
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
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "inputs/arc-gen-repo/tasks")]
from lib import scoring  # noqa: E402
import common  # noqa: E402


SEED = 194_008
CASE_ONE_BASED = 89
AUTHORITY = HERE / "authority/task008.onnx"
AUTHORITY_SHA = "30abdd1f30f1aa88549edbf22c6e7a4af4fec3036fd8809812456ccb0df6e292"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session(disable: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(AUTHORITY)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def main() -> int:
    ort.set_default_logger_severity(4)
    if digest(AUTHORITY) != AUTHORITY_SHA:
        raise RuntimeError("authority drift")
    task = importlib.import_module("task_05f2a901")
    original_randint = common.randint
    original_nibbles = common.rectangle_nibbles
    state: dict[str, Any] = {"case": 0, "outer_randints": [], "nibbles": []}

    def wrapped_randint(start: int, stop: int) -> int:
        value = original_randint(start, stop)
        caller = inspect.currentframe().f_back  # type: ignore[union-attr]
        filename = caller.f_code.co_filename if caller is not None else ""
        if state["case"] == CASE_ONE_BASED and filename.endswith("task_05f2a901.py"):
            state["outer_randints"].append([start, stop, value])
        return value

    def wrapped_nibbles(width: int, height: int, offset: int):
        rows, cols = original_nibbles(width, height, offset)
        if state["case"] == CASE_ONE_BASED:
            state["nibbles"].append({
                "wide": width,
                "tall": height,
                "offset": offset,
                "rows": list(rows),
                "cols": list(cols),
            })
        return rows, cols

    common.randint = wrapped_randint
    common.rectangle_nibbles = wrapped_nibbles
    try:
        random.seed(SEED)
        example = None
        for case in range(1, CASE_ONE_BASED + 1):
            state["case"] = case
            if case == CASE_ONE_BASED:
                state["outer_randints"] = []
                state["nibbles"] = []
            example = task.generate()
    finally:
        common.randint = original_randint
        common.rectangle_nibbles = original_nibbles
    assert example is not None

    values = [event[2] for event in state["outer_randints"]]
    width, height, wide, tall = values[:4]
    redcol, cyancol, flip, xpose = values[-4:]
    placement = values[4:-4]
    if len(placement) % 2:
        raise AssertionError(state["outer_randints"])
    redrow, cyanrow = placement[-2:]
    nibble = state["nibbles"][-1]
    latent = {
        "width": width,
        "height": height,
        "wide": wide,
        "tall": tall,
        "redrow": redrow,
        "redcol": redcol,
        "cyanrow": cyanrow,
        "cyancol": cyancol,
        "rows": nibble["rows"],
        "cols": nibble["cols"],
        "flip": flip,
        "xpose": xpose,
    }
    explicit = task.generate(**latent)
    if explicit != example:
        raise AssertionError({"latent": latent, "events": state})

    benchmark = scoring.convert_to_numpy(example)
    if benchmark is None:
        raise RuntimeError("conversion failed")
    expected = benchmark["output"] > 0
    modes = {}
    for disable, label in ((True, "disable_all"), (False, "default")):
        current = session(disable)
        raw = np.asarray(
            current.run(
                [current.get_outputs()[0].name],
                {current.get_inputs()[0].name: benchmark["input"]},
            )[0]
        )
        decoded = raw > 0
        diff = np.argwhere(decoded != expected)
        modes[label] = {
            "matches_gold": bool(np.array_equal(decoded, expected)),
            "different_onehot_cells": int(diff.shape[0]),
            "runtime_shape": list(raw.shape),
            "declared_shape": current.get_outputs()[0].shape,
            "nonfinite_values": int(raw.size - np.count_nonzero(np.isfinite(raw))),
            "first_differences": [
                {
                    "index": [int(value) for value in index],
                    "expected": bool(expected[tuple(index)]),
                    "predicted": bool(decoded[tuple(index)]),
                    "raw": int(raw[tuple(index)]),
                }
                for index in diff[:40]
            ],
        }

    input_height = len(example["input"])
    input_width = len(example["input"][0])
    disable_session = session(True)
    raw = np.asarray(
        disable_session.run(
            [disable_session.get_outputs()[0].name],
            {disable_session.get_inputs()[0].name: benchmark["input"]},
        )[0]
    )
    predicted_grid = np.argmax(raw[0, :, :input_height, :input_width] > 0, axis=0)
    result = {
        "task": 8,
        "generator": "inputs/arc-gen-repo/tasks/task_05f2a901.py",
        "generator_sha256": digest(ROOT / "inputs/arc-gen-repo/tasks/task_05f2a901.py"),
        "authority_sha256": AUTHORITY_SHA,
        "seed": SEED,
        "case_one_based": CASE_ONE_BASED,
        "latent": latent,
        "explicit_generate_reproduces_seed_case": True,
        "input_grid": example["input"],
        "gold_output_grid": example["output"],
        "authority_decoded_grid": predicted_grid.tolist(),
        "modes": modes,
        "disposition": "AUTHORITY_IS_NOT_GENERATOR_TOTAL_AND_IS_SHAPE_UNTRUTHFUL",
    }
    (HERE / "counterexample.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "latent": latent,
        "disable_diff": modes["disable_all"]["different_onehot_cells"],
        "default_diff": modes["default"]["different_onehot_cells"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
