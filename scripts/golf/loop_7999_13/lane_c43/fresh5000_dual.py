#!/usr/bin/env python3
"""Independent dual-ORT fresh-5000 audit for all task201 archive leads."""

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


TASK_HASH = "846bdb03"
COUNT = 5000
SEED = 800_263_201_043
MODELS = {
    "baseline": HERE / "baseline" / "task201.onnx",
    "r01": HERE / "archive" / "task201_r01_static543.onnx",
    "r02": HERE / "archive" / "task201_r02_static674.onnx",
    "r03": HERE / "archive" / "task201_r03_static785.onnx",
}


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
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


def main() -> int:
    ort.set_default_logger_severity(4)
    generator = importlib.import_module(f"task_{TASK_HASH}")
    random.seed(SEED)
    examples: list[dict[str, np.ndarray]] = []
    generation_errors = 0
    while len(examples) < COUNT:
        try:
            benchmark = scoring.convert_to_numpy(generator.generate())
            if benchmark is not None:
                examples.append(benchmark)
        except Exception:  # noqa: BLE001
            generation_errors += 1

    rows: list[dict[str, object]] = []
    for label, path in MODELS.items():
        model = onnx.load(path)
        row: dict[str, object] = {
            "label": label,
            "path": str(path.relative_to(ROOT)),
            "count": COUNT,
            "seed": SEED,
            "generation_errors": generation_errors,
            "modes": {},
        }
        for disabled, mode in ((True, "disable_all"), (False, "default")):
            runner = make_session(model, disabled)
            result = {
                "right": 0,
                "wrong": 0,
                "runtime_errors": 0,
            }
            for benchmark in examples:
                try:
                    raw = runner.run(["output"], {"input": benchmark["input"]})[0]
                    correct = np.array_equal(raw > 0, benchmark["output"] > 0)
                    result["right"] += int(correct)
                    result["wrong"] += int(not correct)
                except Exception:  # noqa: BLE001
                    result["runtime_errors"] += 1
            result["accuracy"] = result["right"] / COUNT
            result["at_least_95_percent"] = bool(
                result["accuracy"] >= 0.95 and result["runtime_errors"] == 0
            )
            row["modes"][mode] = result
            print(label, mode, json.dumps(result), flush=True)
        row["pass_95_dual"] = bool(
            generation_errors == 0
            and all(result["at_least_95_percent"] for result in row["modes"].values())
        )
        rows.append(row)
    output = HERE / "fresh5000_dual.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
