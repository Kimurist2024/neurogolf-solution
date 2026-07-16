#!/usr/bin/env python3
"""Two independent 5000-case generator audits in default and DISABLE_ALL ORT."""

from __future__ import annotations

import collections
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
COUNT = 5000
TASK_HASHES = {319: "ce602527", 367: "e73095fd"}
SEEDS = {319: (107_319_001, 107_319_002), 367: (107_367_001, 107_367_002)}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def raw_p319(grid: list[list[int]]) -> list[list[int]]:
    """Readable equivalent of inputs/sakana-gcg-2025/raw/task319.py::p."""
    flat = sum(grid, [])
    counts = collections.Counter(flat)
    background = max(flat, key=counts.__getitem__)
    width = len(grid[0])
    coordinates = {
        color: [divmod(index, width) for index, value in enumerate(flat) if value == color]
        for color in set(flat) - {background}
    }
    magnified_color = max(coordinates, key=counts.__getitem__)

    def match_key(color: int) -> tuple[int, int]:
        offsets = collections.Counter(
            (
                mag_row - 2 * small_row - (subpixel % 2),
                mag_col - 2 * small_col - (subpixel > 1),
            )
            for mag_row, mag_col in coordinates[magnified_color]
            for small_row, small_col in coordinates[color]
            for subpixel in range(4)
        )
        return max(offsets.values()) - 2 * counts[color], counts[color]

    target = max(coordinates, key=match_key)
    rows, cols = zip(*coordinates[target])
    return [
        [target if value == target else background for value in row[min(cols): max(cols) + 1]]
        for row in grid[min(rows): max(rows) + 1]
    ]


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


def main() -> int:
    manifest = json.loads((HERE / "audit/build_manifest.json").read_text())
    by_label = {(int(row["task"]), row["label"]): ROOT / row["path"] for row in manifest["candidates"]}
    models = {
        319: {"authority": by_label[(319, "authority")]},
        367: {
            "authority": by_label[(367, "authority")],
            "truthful_true_rule_control": by_label[(367, "truthful_true_rule_control")],
            "truthful_exact_dedupe": by_label[(367, "truthful_exact_dedupe")],
        },
    }
    selected_tasks = set(int(item) for item in sys.argv[1:]) or set(TASK_HASHES)
    configs = ((True, "disable_all"), (False, "default"))
    sessions: dict[tuple[int, str, str], ort.InferenceSession] = {}
    session_errors: dict[str, str] = {}
    for task, items in models.items():
        if task not in selected_tasks:
            continue
        for label, path in items.items():
            for disabled, mode in configs:
                key = (task, label, mode)
                try:
                    sessions[key] = make_session(path, disabled)
                except Exception as exc:  # noqa: BLE001
                    session_errors[f"{task}:{label}:{mode}"] = f"{type(exc).__name__}: {exc}"

    output_path = HERE / "audit/fresh_two_seed.json"
    existing = json.loads(output_path.read_text()) if output_path.exists() else {}
    task_reports: dict[str, Any] = dict(existing.get("tasks", {}))
    for task, task_hash in TASK_HASHES.items():
        if task not in selected_tasks:
            continue
        generator = importlib.import_module(f"task_{task_hash}")
        runs = []
        for seed in SEEDS[task]:
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            stats = {
                label: {
                    mode: {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
                    for _, mode in configs
                }
                for label in models[task]
            }
            reference319 = (
                {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
                if task == 319 else None
            )
            generation_errors = conversion_skips = 0
            valid = 0
            while valid < COUNT:
                try:
                    example = generator.generate()
                except Exception as exc:  # noqa: BLE001
                    generation_errors += 1
                    continue
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    conversion_skips += 1
                    continue
                valid += 1
                expected = benchmark["output"] > 0
                if reference319 is not None:
                    try:
                        got = raw_p319(example["input"])
                        if got == example["output"]:
                            reference319["right"] += 1
                        else:
                            reference319["wrong"] += 1
                            if reference319["first_failure"] is None:
                                reference319["first_failure"] = {
                                    "valid_case": valid,
                                    "input_shape": list(np.asarray(example["input"]).shape),
                                    "expected_shape": list(np.asarray(example["output"]).shape),
                                    "actual_shape": list(np.asarray(got).shape),
                                }
                    except Exception as exc:  # noqa: BLE001
                        reference319["errors"] += 1
                        if reference319["first_failure"] is None:
                            reference319["first_failure"] = {
                                "valid_case": valid,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                for label in models[task]:
                    for _, mode in configs:
                        item = stats[label][mode]
                        session = sessions.get((task, label, mode))
                        if session is None:
                            continue
                        try:
                            raw = session.run(
                                [session.get_outputs()[0].name],
                                {session.get_inputs()[0].name: benchmark["input"]},
                            )[0]
                            if np.array_equal(raw > 0, expected):
                                item["right"] += 1
                            else:
                                item["wrong"] += 1
                                if item["first_failure"] is None:
                                    item["first_failure"] = {
                                        "valid_case": valid,
                                        "input_shape": list(np.asarray(example["input"]).shape),
                                        "output_shape": list(np.asarray(example["output"]).shape),
                                        "differing_elements": int(np.count_nonzero((raw > 0) != expected)),
                                    }
                        except Exception as exc:  # noqa: BLE001
                            item["errors"] += 1
                            if item["first_failure"] is None:
                                item["first_failure"] = {
                                    "valid_case": valid,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
            runs.append({
                "seed": seed,
                "valid": valid,
                "generation_errors": generation_errors,
                "conversion_skips": conversion_skips,
                "models": stats,
                "readable_raw_p_reference": reference319,
            })
            print(
                f"FRESH task{task:03d} seed={seed} "
                f"models={json.dumps(stats, separators=(',', ':'))} "
                f"raw_p={reference319}",
                flush=True,
            )
        task_reports[str(task)] = {
            "hash": task_hash,
            "count_per_seed": COUNT,
            "seeds": list(SEEDS[task]),
            "runs": runs,
        }

    merged_errors = dict(existing.get("session_errors", {}))
    merged_errors.update(session_errors)
    report = {
        "configs": [mode for _, mode in configs],
        "session_errors": merged_errors,
        "tasks": task_reports,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
