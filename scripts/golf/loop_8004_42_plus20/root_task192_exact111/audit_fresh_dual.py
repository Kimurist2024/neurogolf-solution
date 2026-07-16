#!/usr/bin/env python3
"""Independent two-seed task192 fresh audit in both ORT modes."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = HERE / "task192_selected_masks.onnx"
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_sound192_344_93/audit_task192_exact_poly.py"
SEEDS = (192800661, 192930007)
COUNT = 5000

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def load_rule():
    spec = importlib.util.spec_from_file_location("root111_rule", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.task192_rule


def make_session(model: onnx.ModelProto, disable: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def main() -> int:
    rule = load_rule()
    generator = importlib.import_module("task_7e0986d6")
    model = onnx.load(CANDIDATE)
    sessions = {
        "disable_all": make_session(model, True),
        "default": make_session(model, False),
    }
    rows = []
    for seed in SEEDS:
        random.seed(seed)
        row = {
            "seed": seed,
            "total": COUNT,
            "reference_right": 0,
            "modes": {
                name: {
                    "right": 0,
                    "runtime_errors": 0,
                    "nonfinite_values": 0,
                    "first_failure": None,
                }
                for name in sessions
            },
            "cross_mode_threshold_equal": 0,
        }
        for index in range(COUNT):
            example = generator.generate()
            row["reference_right"] += int(rule(example["input"]) == example["output"])
            benchmark = scoring.convert_to_numpy(example)
            outputs: dict[str, np.ndarray] = {}
            if benchmark is None:
                for result in row["modes"].values():
                    result["runtime_errors"] += 1
                    result["first_failure"] = result["first_failure"] or {
                        "index": index, "error": "convert_to_numpy returned None"
                    }
                continue
            expected = benchmark["output"].astype(bool)
            for name, session in sessions.items():
                result = row["modes"][name]
                try:
                    raw = np.asarray(session.run(None, {"input": benchmark["input"]})[0])
                except Exception as exc:  # noqa: BLE001
                    result["runtime_errors"] += 1
                    result["first_failure"] = result["first_failure"] or {
                        "index": index, "error": f"{type(exc).__name__}: {exc}"
                    }
                    continue
                result["nonfinite_values"] += int(raw.size - np.count_nonzero(np.isfinite(raw)))
                correct = np.array_equal(raw > 0, expected)
                result["right"] += int(correct)
                if not correct and result["first_failure"] is None:
                    result["first_failure"] = {
                        "index": index,
                        "different_cells": int(np.count_nonzero((raw > 0) != expected)),
                    }
                outputs[name] = raw
            if len(outputs) == 2:
                row["cross_mode_threshold_equal"] += int(
                    np.array_equal(outputs["disable_all"] > 0, outputs["default"] > 0)
                )
        for result in row["modes"].values():
            result["perfect"] = (
                result["right"] == COUNT
                and result["runtime_errors"] == 0
                and result["nonfinite_values"] == 0
            )
        row["perfect"] = (
            row["reference_right"] == COUNT
            and row["cross_mode_threshold_equal"] == COUNT
            and all(result["perfect"] for result in row["modes"].values())
        )
        rows.append(row)
        print(
            f"seed={seed} reference={row['reference_right']}/{COUNT} "
            f"disable={row['modes']['disable_all']['right']}/{COUNT} "
            f"default={row['modes']['default']['right']}/{COUNT} "
            f"perfect={row['perfect']}",
            flush=True,
        )
    report = {"candidate": str(CANDIDATE.relative_to(ROOT)), "rows": rows}
    report["accepted"] = all(row["perfect"] for row in rows)
    (HERE / "fresh_dual.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
