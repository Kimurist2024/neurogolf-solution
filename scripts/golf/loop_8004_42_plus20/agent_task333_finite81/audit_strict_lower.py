#!/usr/bin/env python3
"""Four-configuration known audit for every unique strict-lower task333 SHA."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def session(data: bytes, disable: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_from_string(data)))
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def stats() -> dict:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "near_positive_values": 0,
        "min_positive": None,
        "max_abs_raw": 0.0,
        "output_shapes": [],
        "first_failure": None,
    }


def main() -> None:
    inventory = json.loads((HERE / "candidate_inventory.json").read_text())
    examples = scoring.load_examples(333)
    candidates = [row for row in inventory["rows"] if row["strictly_lower_than_8005_17"]]
    rows = []
    for candidate in candidates:
        data = (ROOT / candidate["extracted_path"]).read_bytes()
        configs = {}
        for disable, threads, label in CONFIGS:
            row = stats()
            try:
                sess = session(data, disable, threads)
            except Exception as exc:  # noqa: BLE001
                row["session_error"] = f"{type(exc).__name__}: {exc}"
                row["runtime_errors"] = 265
                configs[label] = row
                continue
            for split in ("train", "test", "arc-gen"):
                for index, example in enumerate(examples[split]):
                    benchmark = scoring.convert_to_numpy(example)
                    if benchmark is None:
                        continue
                    try:
                        raw = sess.run(
                            [sess.get_outputs()[0].name],
                            {sess.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                        finite = np.isfinite(raw)
                        row["nonfinite_values"] += int(raw.size - np.count_nonzero(finite))
                        safe = raw[finite]
                        if safe.size:
                            positive = safe[safe > 0]
                            row["near_positive_values"] += int(np.count_nonzero((safe > 0) & (safe < 0.25)))
                            if positive.size:
                                value = float(positive.min())
                                row["min_positive"] = value if row["min_positive"] is None else min(row["min_positive"], value)
                            row["max_abs_raw"] = max(row["max_abs_raw"], float(np.abs(safe).max(initial=0.0)))
                        shape = list(raw.shape)
                        if shape not in row["output_shapes"]:
                            row["output_shapes"].append(shape)
                        expected = benchmark["output"].astype(bool)
                        if np.array_equal(raw > 0, expected):
                            row["right"] += 1
                        else:
                            row["wrong"] += 1
                            if row["first_failure"] is None:
                                row["first_failure"] = {
                                    "split": split,
                                    "index": index,
                                    "different_cells": int(np.count_nonzero((raw > 0) != expected)),
                                }
                    except Exception as exc:  # noqa: BLE001
                        row["runtime_errors"] += 1
                        row["first_failure"] = row["first_failure"] or {
                            "split": split,
                            "index": index,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
            row["total"] = row["right"] + row["wrong"] + row["runtime_errors"]
            row["perfect"] = (
                row["right"] == row["total"]
                and row["wrong"] == row["runtime_errors"] == row["nonfinite_values"] == row["near_positive_values"] == 0
            )
            configs[label] = row
        shape_truthful = all(
            config.get("output_shapes") == [[1, 10, 30, 30]] for config in configs.values()
        )
        rows.append(
            {
                **candidate,
                "known_four_configs": configs,
                "known_perfect_all_configs": all(row.get("perfect", False) for row in configs.values()),
                "runtime_output_shape_truthful": shape_truthful,
            }
        )
        print(candidate["sha256"][:12], candidate["profile"]["cost"], rows[-1]["known_perfect_all_configs"], [row["right"] for row in configs.values()], flush=True)

    result = {
        "baseline_zip": inventory["baseline_zip"],
        "baseline_zip_sha256": inventory["baseline_zip_sha256"],
        "baseline_profile": inventory["baseline_profile"],
        "candidate_count": len(rows),
        "known_perfect_count": sum(row["known_perfect_all_configs"] for row in rows),
        "rows": rows,
    }
    (HERE / "strict_lower_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
