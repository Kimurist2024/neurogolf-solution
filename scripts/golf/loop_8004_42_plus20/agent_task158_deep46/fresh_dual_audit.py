#!/usr/bin/env python3
"""Dual-ORT independent fresh audit for the task158 priority models."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
sys.path.insert(0, str(HERE))
from lib import scoring  # noqa: E402
from reference_task158 import solve_grid  # noqa: E402


MODELS = {
    "current_7615": HERE / "baseline/task158.onnx",
    "spec_anchor13_7657": ROOT / "scripts/golf/loop_7999_13/lane_a22/sound/task158_spec_anchor13.onnx",
    "archive_r01_sha3728_actual7838": ROOT / "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task158_r01_static1844.onnx",
    "archive_r02_shaf1d5_actual7886": ROOT / "scripts/golf/loop_7999_13/lane_archive_loose_sweep/task158_r02_static1860.onnx",
}
SEEDS = (158_046_3, 158_046_4)
COUNT_PER_SEED = 3000


def make_session(path: Path, disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def empty_stats() -> dict[str, object]:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "reference_agreement": 0,
        "nonfinite_values": 0,
        "small_positive_values_0_to_0p25": 0,
        "min_positive_raw": None,
        "max_nonpositive_raw": None,
        "first_failure": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--count", type=int, default=COUNT_PER_SEED)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--output", type=Path, default=HERE / "evidence/fresh_dual_priority.json"
    )
    args = parser.parse_args()
    models = MODELS
    if args.winner_only:
        models = {
            "current_7615": HERE / "baseline/task158.onnx",
            "candidate_7612": HERE / "sound/task158_scatter_max_orientation_only.onnx",
        }
    ort.set_default_logger_severity(4)
    generator = importlib.import_module("task_6aa20dc0")
    modes = ((True, "disable_all"), (False, "default"))
    sessions: dict[str, dict[str, ort.InferenceSession]] = {}
    session_errors: dict[str, dict[str, str]] = {}
    for label, path in models.items():
        sessions[label] = {}
        session_errors[label] = {}
        for disabled, mode in modes:
            try:
                sessions[label][mode] = make_session(path, disabled)
            except Exception as exc:  # noqa: BLE001
                session_errors[label][mode] = f"{type(exc).__name__}: {exc}"

    all_seed_rows = []
    started = time.monotonic()
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        stats = {
            label: {mode: empty_stats() for _, mode in modes}
            for label in models
        }
        reference = {
            "right": 0,
            "wrong": 0,
            "first_failure": None,
        }
        shapes: Counter[str] = Counter()
        for index in range(1, args.count + 1):
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                raise RuntimeError("generator example conversion failed")
            expected = benchmark["output"] > 0
            reference_grid = solve_grid(example["input"])
            reference_expected = np.zeros_like(expected, dtype=bool)
            height, width = reference_grid.shape
            for colour in range(10):
                reference_expected[0, colour, :height, :width] = reference_grid == colour
            reference_ok = np.array_equal(reference_expected, expected)
            if reference_ok:
                reference["right"] += 1
            else:
                reference["wrong"] += 1
                if reference["first_failure"] is None:
                    reference["first_failure"] = {
                        "case": index,
                        "different_cells": int(np.count_nonzero(reference_expected != expected)),
                    }
            shapes[f"{height}x{width}"] += 1

            for label in models:
                for _, mode in modes:
                    row = stats[label][mode]
                    session = sessions[label].get(mode)
                    if session is None:
                        row["runtime_errors"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "case": index,
                                "stage": "session",
                                "error": session_errors[label][mode],
                            }
                        continue
                    try:
                        raw = session.run(
                            [session.get_outputs()[0].name],
                            {session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                        decoded = raw > 0
                        if np.array_equal(decoded, expected):
                            row["right"] += 1
                        else:
                            row["wrong"] += 1
                            if row["first_failure"] is None:
                                row["first_failure"] = {
                                    "case": index,
                                    "stage": "gold",
                                    "different_cells": int(np.count_nonzero(decoded != expected)),
                                    "shape": f"{height}x{width}",
                                }
                        row["reference_agreement"] += int(
                            np.array_equal(decoded, reference_expected)
                        )
                        finite = np.isfinite(raw)
                        row["nonfinite_values"] += int(np.count_nonzero(~finite))
                        positive = raw[np.logical_and(finite, raw > 0)]
                        nonpositive = raw[np.logical_and(finite, raw <= 0)]
                        row["small_positive_values_0_to_0p25"] += int(
                            np.count_nonzero(np.logical_and(positive > 0, positive < 0.25))
                        )
                        if positive.size:
                            value = float(np.min(positive))
                            old = row["min_positive_raw"]
                            row["min_positive_raw"] = value if old is None else min(old, value)
                        if nonpositive.size:
                            value = float(np.max(nonpositive))
                            old = row["max_nonpositive_raw"]
                            row["max_nonpositive_raw"] = value if old is None else max(old, value)
                    except Exception as exc:  # noqa: BLE001
                        row["runtime_errors"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "case": index,
                                "stage": "runtime",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
            if index % 500 == 0:
                print(f"seed={seed} {index}/{args.count}", flush=True)

        passed_models = {}
        for label in models:
            passed_models[label] = all(
                stats[label][mode]["right"] == args.count
                and stats[label][mode]["wrong"] == 0
                and stats[label][mode]["runtime_errors"] == 0
                and stats[label][mode]["reference_agreement"] == args.count
                and stats[label][mode]["nonfinite_values"] == 0
                and stats[label][mode]["small_positive_values_0_to_0p25"] == 0
                for _, mode in modes
            )
        seed_row = {
            "seed": seed,
            "requested": args.count,
            "shapes": dict(sorted(shapes.items())),
            "reference": reference,
            "models": stats,
            "passed_models": passed_models,
        }
        all_seed_rows.append(seed_row)
        args.output.write_text(
            json.dumps({"seeds": all_seed_rows, "complete": False}, indent=2) + "\n"
        )

    final_pass = {
        label: all(row["passed_models"][label] for row in all_seed_rows)
        for label in models
    }
    result = {
        "task": 158,
        "task_hash": "6aa20dc0",
        "count_per_seed": args.count,
        "seeds": all_seed_rows,
        "session_errors": session_errors,
        "passed_models": final_pass,
        "complete": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({"passed_models": final_pass, "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    return 0 if all(final_pass.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
