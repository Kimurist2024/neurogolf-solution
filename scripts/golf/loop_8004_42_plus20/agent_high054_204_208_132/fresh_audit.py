#!/usr/bin/env python3
"""Two-seed fresh audit in disable-all and default ORT for lane 132 baselines."""

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
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402

HASHES = {54: "264363fd", 204: "868de0fa", 208: "890034e9"}
PER_SEED = 1500


def make_session(data: bytes, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(onnx.load_from_string(data))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def main() -> None:
    ort.set_default_logger_severity(4)
    result = {
        "fresh_per_seed": PER_SEED,
        "note": "default-ORT session failures are recorded rather than bypassed",
        "tasks": {},
    }
    for task, hash8 in HASHES.items():
        data = (HERE / f"baseline/task{task:03d}.onnx").read_bytes()
        sessions, session_errors = {}, {}
        for disable_all, mode in ((True, "disable_all"), (False, "default")):
            try:
                sessions[mode] = make_session(data, disable_all)
            except Exception as exc:  # noqa: BLE001
                session_errors[mode] = f"{type(exc).__name__}: {exc}"
        task_result = {"session_errors": session_errors, "seeds": []}
        generator = importlib.import_module(f"task_{hash8}")
        for seed in (132_000_000 + task, 132_100_000 + task):
            random.seed(seed)
            stats = {
                mode: {
                    "right": 0,
                    "wrong": 0,
                    "errors": 0,
                    "nonfinite": 0,
                    "near_positive": 0,
                    "min_positive": None,
                    "first_failure": None,
                }
                for mode in sessions
            }
            generated = 0
            generation_errors = 0
            while generated < PER_SEED:
                try:
                    benchmark = scoring.convert_to_numpy(generator.generate())
                except Exception:  # noqa: BLE001
                    generation_errors += 1
                    continue
                if benchmark is None:
                    continue
                generated += 1
                for mode, session in sessions.items():
                    row = stats[mode]
                    try:
                        raw = np.asarray(
                            session.run(
                                [session.get_outputs()[0].name],
                                {session.get_inputs()[0].name: benchmark["input"]},
                            )[0]
                        )
                        correct = np.array_equal(raw > 0, benchmark["output"] > 0)
                        row["right"] += int(correct)
                        row["wrong"] += int(not correct)
                        row["nonfinite"] += int(not np.isfinite(raw).all())
                        positives = raw[raw > 0]
                        if positives.size:
                            minimum = float(positives.min())
                            row["min_positive"] = (
                                minimum
                                if row["min_positive"] is None
                                else min(row["min_positive"], minimum)
                            )
                            row["near_positive"] += int(
                                np.count_nonzero((raw > 0) & (raw < 0.25))
                            )
                        if not correct and row["first_failure"] is None:
                            row["first_failure"] = {"case": generated, "kind": "wrong"}
                    except Exception as exc:  # noqa: BLE001
                        row["errors"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "case": generated,
                                "kind": "runtime_error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
            task_result["seeds"].append(
                {
                    "seed": seed,
                    "generated": generated,
                    "generation_errors": generation_errors,
                    "modes": stats,
                }
            )
        result["tasks"][str(task)] = task_result
        print(task, task_result, flush=True)
    (HERE / "audit/fresh_two_seed.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
