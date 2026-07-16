#!/usr/bin/env python3
"""Fresh two-seed proof for the cheapest prior truthful task216 control."""

from __future__ import annotations

import copy
import importlib
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
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(HERE))

from fresh_two_seeds import COUNT, SEEDS, encode, fresh_216  # noqa: E402
from lib import scoring  # noqa: E402


MODEL = ROOT / "scripts/golf/scratch/task216/cand3.onnx"


def session(mode: str) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(MODEL)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options)


def main() -> int:
    generator = importlib.import_module("task_8efcae92")
    sessions = {mode: session(mode) for mode in ("disabled", "default")}
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rng = random.Random(seed + 216)
        mode_rows = {
            mode: {
                "task": 216,
                "seed": seed,
                "count": COUNT,
                "mode": mode,
                "right": 0,
                "wrong": 0,
                "errors": 0,
                "nonfinite": 0,
                "mid_margin": 0,
                "min_positive": None,
                "first_failure": None,
            }
            for mode in sessions
        }
        for index in range(COUNT):
            example = fresh_216(generator, rng, index)
            input_value = encode(example["input"])
            expected = encode(example["output"]).astype(bool)
            for mode, runner in sessions.items():
                row = mode_rows[mode]
                try:
                    raw = np.asarray(runner.run(["output"], {"input": input_value})[0])
                    finite = np.isfinite(raw)
                    row["nonfinite"] += int(np.count_nonzero(~finite))
                    row["mid_margin"] += int(
                        np.count_nonzero(finite & (raw > 0.0) & (raw < 0.25))
                    )
                    positive = raw[finite & (raw > 0.0)]
                    if positive.size:
                        current = float(positive.min())
                        prior = row["min_positive"]
                        row["min_positive"] = current if prior is None else min(prior, current)
                    if np.array_equal(raw > 0.0, expected):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        row["first_failure"] = row["first_failure"] or {
                            "index": index,
                            "kind": "wrong_output",
                        }
                except Exception as exc:  # noqa: BLE001
                    row["errors"] += 1
                    row["first_failure"] = row["first_failure"] or {
                        "index": index,
                        "kind": "runtime_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            if (index + 1) % 1000 == 0:
                print(seed, "progress", index + 1, flush=True)
        for row in mode_rows.values():
            row["pass_exact"] = (
                row["right"] == COUNT
                and row["wrong"] == 0
                and row["errors"] == 0
                and row["nonfinite"] == 0
                and row["mid_margin"] == 0
            )
            rows.append(row)
            print(row, flush=True)
    report = {
        "model": str(MODEL.relative_to(ROOT)),
        "purpose": "truthful exact control, not a strict-lower candidate",
        "count_per_seed": COUNT,
        "seeds": list(SEEDS),
        "rows": rows,
    }
    (HERE / "fresh_truthful216.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
