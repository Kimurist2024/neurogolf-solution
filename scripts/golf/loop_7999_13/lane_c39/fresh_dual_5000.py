#!/usr/bin/env python3
"""Run task343 baseline/candidate on one identical fresh corpus in both ORT modes."""

from __future__ import annotations

import copy
import hashlib
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))

from lib import scoring  # noqa: E402


TASK = 343
COUNT = 5000
# Reproduce the archived r01 4975/5000 corpus exactly, while applying the same
# stream to the baseline and collecting runtime errors separately from wrongs.
SEED = 343_799_445
MODELS = {
    "baseline": HERE / "baseline" / "task343.onnx",
    "candidate": HERE / "candidate" / "task343.onnx",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_session(path: Path, mode: str) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError(f"sanitize failed: {path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if mode == "disable_all":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def empty_counter(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "first_wrong": None,
        "first_runtime_error": None,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    generator = importlib.import_module(f"task_{task_map[f'{TASK:03d}']}")
    sessions = {
        (label, mode): make_session(path, mode)
        for label, path in MODELS.items()
        for mode in ("disable_all", "default")
    }
    results = {
        label: {mode: empty_counter(path) for mode in ("disable_all", "default")}
        for label, path in MODELS.items()
    }
    comparison = {
        mode: {
            "both_correct": 0,
            "baseline_only_correct": 0,
            "candidate_only_correct": 0,
            "both_wrong": 0,
            "threshold_outputs_equal": 0,
            "threshold_outputs_different": 0,
        }
        for mode in ("disable_all", "default")
    }
    mode_parity = {
        label: {"equal": 0, "different": 0, "not_comparable_due_to_error": 0}
        for label in MODELS
    }
    generated = generation_errors = attempts = 0
    random.seed(SEED)
    while generated < COUNT:
        attempts += 1
        try:
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
        except Exception:  # noqa: BLE001 - counted separately in the evidence
            generation_errors += 1
            continue
        if benchmark is None:
            generation_errors += 1
            continue
        case_index = generated
        grid_shape = [len(example["input"]), len(example["input"][0])]
        generated += 1
        gold = benchmark["output"] > 0.0
        outputs: dict[tuple[str, str], np.ndarray | None] = {}
        correct: dict[tuple[str, str], bool | None] = {}
        for key, session in sessions.items():
            label, mode = key
            row = results[label][mode]
            try:
                raw = session.run(["output"], {"input": benchmark["input"]})[0]
                threshold = np.asarray(raw) > 0.0
                outputs[key] = threshold
                is_correct = bool(np.array_equal(threshold, gold))
                correct[key] = is_correct
                if is_correct:
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    if row["first_wrong"] is None:
                        row["first_wrong"] = {
                            "case": case_index,
                            "grid_shape": grid_shape,
                            "different_threshold_cells": int(np.count_nonzero(threshold != gold)),
                        }
            except Exception as exc:  # noqa: BLE001 - runtime errors are a separate gate
                outputs[key] = None
                correct[key] = None
                row["runtime_errors"] += 1
                if row["first_runtime_error"] is None:
                    row["first_runtime_error"] = {"case": case_index, "error": repr(exc)}

        for mode in ("disable_all", "default"):
            base_ok = correct[("baseline", mode)]
            cand_ok = correct[("candidate", mode)]
            if base_ok is not None and cand_ok is not None:
                if base_ok and cand_ok:
                    comparison[mode]["both_correct"] += 1
                elif base_ok:
                    comparison[mode]["baseline_only_correct"] += 1
                elif cand_ok:
                    comparison[mode]["candidate_only_correct"] += 1
                else:
                    comparison[mode]["both_wrong"] += 1
                same = np.array_equal(
                    outputs[("baseline", mode)], outputs[("candidate", mode)]
                )
                comparison[mode][
                    "threshold_outputs_equal" if same else "threshold_outputs_different"
                ] += 1
        for label in MODELS:
            left = outputs[(label, "disable_all")]
            right = outputs[(label, "default")]
            if left is None or right is None:
                mode_parity[label]["not_comparable_due_to_error"] += 1
            elif np.array_equal(left, right):
                mode_parity[label]["equal"] += 1
            else:
                mode_parity[label]["different"] += 1

    for label in MODELS:
        for mode in ("disable_all", "default"):
            row = results[label][mode]
            row["accuracy"] = row["right"] / generated
            row["passes_user_95_percent_gate"] = (
                row["accuracy"] >= 0.95 and row["runtime_errors"] == 0
            )
    report = {
        "task": TASK,
        "generator_hash": task_map[f"{TASK:03d}"],
        "seed": SEED,
        "requested": COUNT,
        "attempts": attempts,
        "generated": generated,
        "generation_errors": generation_errors,
        "results": results,
        "comparison": comparison,
        "ort_mode_parity": mode_parity,
        "candidate_is_95_percent_eligible": all(
            results["candidate"][mode]["passes_user_95_percent_gate"]
            for mode in ("disable_all", "default")
        ),
        "candidate_is_accuracy_non_regressing": all(
            results["candidate"][mode]["right"] >= results["baseline"][mode]["right"]
            for mode in ("disable_all", "default")
        ),
    }
    (HERE / "fresh_dual_5000.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
