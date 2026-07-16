#!/usr/bin/env python3
"""Independent fresh 2x5000 dual-ORT and raw-equivalence audit."""

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
from lib import scoring  # noqa: E402


CANDIDATE = HERE / "sound/task158_exact_repair_cost7529.onnx"
TRUSTED = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task158_deep46/sound/"
    "task158_scatter_max_orientation_only.onnx"
)
DEFAULT_SEEDS = (1_581_081, 1_581_082)


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
        "raw_equal_trusted": 0,
        "raw_unequal_trusted": 0,
        "nonfinite_values": 0,
        "small_positive_values_0_to_0p25": 0,
        "min_positive_raw": None,
        "max_nonpositive_raw": None,
        "max_abs_delta_trusted": 0.0,
        "first_failure": None,
    }


def update_margin(row: dict[str, object], raw: np.ndarray) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output", type=Path, default=HERE / "evidence/fresh_dual_affine_2x5000.json"
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ort.set_default_logger_severity(4)
    generator = importlib.import_module("task_6aa20dc0")
    modes = ((True, "disable_all"), (False, "default"))
    sessions = {
        label: {
            mode: make_session(path, disabled)
            for disabled, mode in modes
        }
        for label, path in (("candidate_7529", CANDIDATE), ("trusted_7612", TRUSTED))
    }

    seed_rows = []
    started = time.monotonic()
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        stats = {mode: empty_stats() for _, mode in modes}
        trusted_stats = {mode: {"right": 0, "wrong": 0, "errors": 0} for _, mode in modes}
        shapes: Counter[str] = Counter()
        generation_errors = 0
        generation_first_failure = None
        for case in range(1, args.count + 1):
            try:
                example = generator.generate()
            except Exception as exc:  # noqa: BLE001
                generation_errors += 1
                if generation_first_failure is None:
                    generation_first_failure = {
                        "case": case,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                continue
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                raise RuntimeError("fresh example conversion failed")
            expected = benchmark["output"] > 0
            height = len(example["output"])
            width = len(example["output"][0])
            shapes[f"{height}x{width}"] += 1
            for _, mode in modes:
                row = stats[mode]
                try:
                    candidate_session = sessions["candidate_7529"][mode]
                    trusted_session = sessions["trusted_7612"][mode]
                    feed_candidate = {
                        candidate_session.get_inputs()[0].name: benchmark["input"]
                    }
                    feed_trusted = {
                        trusted_session.get_inputs()[0].name: benchmark["input"]
                    }
                    raw = candidate_session.run(
                        [candidate_session.get_outputs()[0].name], feed_candidate
                    )[0]
                    raw_trusted = trusted_session.run(
                        [trusted_session.get_outputs()[0].name], feed_trusted
                    )[0]
                    if np.array_equal(raw > 0, expected):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "case": case,
                                "stage": "gold",
                                "different_cells": int(np.count_nonzero((raw > 0) != expected)),
                            }
                    if np.array_equal(raw_trusted > 0, expected):
                        trusted_stats[mode]["right"] += 1
                    else:
                        trusted_stats[mode]["wrong"] += 1
                    delta = float(np.max(np.abs(raw - raw_trusted)))
                    row["max_abs_delta_trusted"] = max(
                        float(row["max_abs_delta_trusted"]), delta
                    )
                    if np.array_equal(raw, raw_trusted):
                        row["raw_equal_trusted"] += 1
                    else:
                        row["raw_unequal_trusted"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "case": case,
                                "stage": "raw_equivalence",
                                "different_values": int(np.count_nonzero(raw != raw_trusted)),
                                "max_abs_delta": delta,
                            }
                    update_margin(row, raw)
                except Exception as exc:  # noqa: BLE001
                    row["runtime_errors"] += 1
                    trusted_stats[mode]["errors"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {
                            "case": case,
                            "stage": "runtime",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
            if case % 250 == 0:
                print(f"seed={seed} {case}/{args.count}", flush=True)

        passed_modes = {
            mode: (
                stats[mode]["right"] == args.count
                and stats[mode]["wrong"] == 0
                and stats[mode]["runtime_errors"] == 0
                and stats[mode]["raw_equal_trusted"] == args.count
                and stats[mode]["raw_unequal_trusted"] == 0
                and stats[mode]["nonfinite_values"] == 0
                and stats[mode]["small_positive_values_0_to_0p25"] == 0
                and trusted_stats[mode]["right"] == args.count
                and trusted_stats[mode]["wrong"] == 0
                and trusted_stats[mode]["errors"] == 0
            )
            for _, mode in modes
        }
        seed_rows.append(
            {
                "seed": seed,
                "requested": args.count,
                "generation_errors": generation_errors,
                "generation_first_failure": generation_first_failure,
                "shapes": dict(sorted(shapes.items())),
                "candidate": stats,
                "trusted": trusted_stats,
                "passed_modes": passed_modes,
                "passed": generation_errors == 0 and all(passed_modes.values()),
            }
        )
        args.output.write_text(
            json.dumps({"seeds": seed_rows, "complete": False}, indent=2) + "\n"
        )

    result = {
        "task": 158,
        "task_hash": "6aa20dc0",
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "trusted": str(TRUSTED.relative_to(ROOT)),
        "count_per_seed": args.count,
        "seeds": seed_rows,
        "passed": all(row["passed"] for row in seed_rows),
        "complete": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
