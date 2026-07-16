#!/usr/bin/env python3
"""Strict static, runtime-shape, dual-ORT and known raw audit for this lane."""

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
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_a20"))
import audit_history as shared  # noqa: E402
from lib import scoring  # noqa: E402


shared.BASE_COST = {158: 7578}
TRUSTED = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task158_deep46/sound/"
    "task158_scatter_max_orientation_only.onnx"
)


def session(path: Path, disable_all: bool) -> ort.InferenceSession:
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


def known_raw_equivalence(path: Path) -> dict[str, object]:
    rows: dict[str, object] = {}
    examples = [
        example
        for subset in scoring.load_examples(158).values()
        for example in subset
    ]
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        try:
            candidate = session(path, disabled)
            trusted = session(TRUSTED, disabled)
        except Exception as exc:  # noqa: BLE001
            rows[mode] = {
                "session_error": f"{type(exc).__name__}: {exc}",
                "right": 0,
                "wrong": 0,
                "errors": len(examples),
            }
            continue
        right = wrong = errors = 0
        max_abs_delta = 0.0
        first_failure = None
        for index, example in enumerate(examples):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                left = candidate.run(
                    [candidate.get_outputs()[0].name],
                    {candidate.get_inputs()[0].name: benchmark["input"]},
                )[0]
                right_raw = trusted.run(
                    [trusted.get_outputs()[0].name],
                    {trusted.get_inputs()[0].name: benchmark["input"]},
                )[0]
                delta = float(np.max(np.abs(left - right_raw)))
                max_abs_delta = max(max_abs_delta, delta)
                if np.array_equal(left, right_raw):
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        first_failure = {
                            "case": index,
                            "different_values": int(np.count_nonzero(left != right_raw)),
                            "max_abs_delta": delta,
                        }
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if first_failure is None:
                    first_failure = {
                        "case": index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        rows[mode] = {
            "right": right,
            "wrong": wrong,
            "errors": errors,
            "max_abs_delta": max_abs_delta,
            "first_failure": first_failure,
        }
    return rows


def main() -> None:
    paths = {
        "baseline_7578": HERE / "baseline/task158.onnx",
        "source_high_only_7570_unsafe_control": (
            HERE / "candidates/task158_source_high_only_unsafe_control.onnx"
        ),
        "invalid_base_repair_7584_control": (
            HERE / "candidates/task158_invalid_base_repair_control.onnx"
        ),
        "invalid_base_repair_source_high_7576": (
            HERE / "candidates/task158_invalid_base_repair_source_high.onnx"
        ),
        "invalid_base_repair_anchor_sum_expected_7552": (
            HERE / "candidates/task158_invalid_base_repair_anchor_sum.onnx"
        ),
        "invalid_base_repair_affine_sum_expected_7529": (
            HERE / "sound/task158_exact_repair_cost7529.onnx"
        ),
    }
    rows = []
    for label, path in paths.items():
        row = shared.audit(
            158,
            label,
            path,
            None,
            ["submission_base_8008.14.zip"],
            baseline=label.startswith("baseline"),
        )
        row["known_raw_equivalence_to_trusted_7612"] = known_raw_equivalence(path)
        rows.append(row)
        (HERE / "evidence/audit.json").write_text(
            json.dumps({"rows": rows, "complete": False}, indent=2) + "\n"
        )
        profile = row.get("official_like_score") or {}
        print(
            label,
            profile.get("cost"),
            row.get("known_disable_all"),
            row.get("known_default"),
            row.get("pre_fresh_reasons"),
            flush=True,
        )
    result = {"rows": rows, "complete": True}
    (HERE / "evidence/audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
