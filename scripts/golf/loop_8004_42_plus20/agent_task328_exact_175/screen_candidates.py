#!/usr/bin/env python3
"""Fast legal-witness screen for task328 exact553 compensation placements."""

from __future__ import annotations

import copy
import importlib
import itertools
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


def make_session(path: Path) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def examples() -> list[tuple[str, dict]]:
    generator = importlib.import_module("task_d22278a0")
    rows: list[tuple[str, dict]] = []
    for case in range(16):
        random.seed(328_260_000 + case)
        rows.append((f"fresh_{case}", generator.generate()))
    # Add one representative for every size and corner count, plus the old
    # high-magnitude opposite-corner witness.
    for size in range(6, 19):
        corners = ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
        for count in range(2, 5):
            selected = tuple(itertools.combinations(corners, count))[0]
            rr, cc = zip(*selected)
            rows.append(
                (
                    f"canonical_s{size}_c{count}",
                    generator.generate(
                        size=size,
                        rows=rr,
                        cols=cc,
                        colors=tuple(range(1, count + 1)),
                    ),
                )
            )
    rows.append(
        (
            "opposite_8",
            generator.generate(
                size=8, rows=(0, 7), cols=(0, 7), colors=(1, 2)
            ),
        )
    )
    return rows


def audit(path: Path, cases: list[tuple[str, dict]]) -> dict:
    session = make_session(path)
    row = {
        "path": str(path.relative_to(ROOT)),
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "near_positive_values": 0,
        "true_below_0_25": 0,
        "false_positive_values": 0,
        "min_true": None,
        "max_abs": 0.0,
        "first_failure": None,
    }
    for label, example in cases:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(label)
        expected = benchmark["output"].astype(bool)
        try:
            raw = session.run(
                [session.get_outputs()[0].name],
                {session.get_inputs()[0].name: benchmark["input"]},
            )[0]
        except Exception as exc:  # noqa: BLE001
            row["runtime_errors"] += 1
            if row["first_failure"] is None:
                row["first_failure"] = {
                    "case": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            continue
        finite = np.isfinite(raw)
        row["nonfinite_values"] += int((~finite).sum())
        if np.any(finite):
            row["max_abs"] = max(
                float(row["max_abs"]), float(np.abs(raw[finite]).max())
            )
        near = finite & (raw > 0) & (raw < 0.25)
        row["near_positive_values"] += int(near.sum())
        true_raw = raw[expected]
        false_raw = raw[~expected]
        row["true_below_0_25"] += int(np.count_nonzero(true_raw < 0.25))
        row["false_positive_values"] += int(np.count_nonzero(false_raw > 0))
        if true_raw.size and np.isfinite(true_raw).all():
            value = float(true_raw.min())
            row["min_true"] = value if row["min_true"] is None else min(row["min_true"], value)
        correct = np.array_equal(raw > 0, expected)
        row["right" if correct else "wrong"] += 1
        if not correct and row["first_failure"] is None:
            row["first_failure"] = {
                "case": label,
                "different_cells": int(np.count_nonzero((raw > 0) != expected)),
            }
    return row


def main() -> None:
    cases = examples()
    paths = [HERE / "controls/authority558.onnx", HERE / "controls/exact554.onnx"]
    paths.extend(
        HERE / "candidates" / name
        for name in (
            "task328_exact553_split_m12.onnx",
            "task328_exact553_split_m6.onnx",
            "task328_exact553_split_p0.onnx",
            "task328_exact553_split_p6.onnx",
            "task328_exact553_split_p12.onnx",
        )
    )
    rows = []
    for path in paths:
        row = audit(path, cases)
        rows.append(row)
        print(
            path.name,
            row["right"],
            row["wrong"],
            "near", row["near_positive_values"],
            "min", row["min_true"],
            flush=True,
        )
    result = {"case_count": len(cases), "config": "disable_all_t1", "rows": rows}
    (HERE / "candidate_screen.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
