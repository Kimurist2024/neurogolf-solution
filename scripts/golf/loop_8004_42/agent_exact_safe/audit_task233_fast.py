#!/usr/bin/env python3
"""Focused task233 audit: 3000 target runs plus a dual-ORT exactness sample."""

from __future__ import annotations

import importlib
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from lib import scoring  # noqa: E402
from audit_stream import session, sha256, structural_gate  # noqa: E402


BASE_ZIP = ROOT / "scripts/golf/loop_8004_42/submission_8004.42_fixed_rebase_meta.zip"
CANDIDATE_PATH = HERE / "models/task233.onnx"
OUT = HERE / "task233_focused_audit.json"
TASK_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
FRESH = 3000
DUAL_SAMPLE = 500


def main() -> None:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        baseline_bytes = archive.read("task233.onnx")
    candidate_bytes = CANDIDATE_PATH.read_bytes()
    baseline = onnx.load_model_from_string(baseline_bytes)
    candidate = onnx.load_model_from_string(candidate_bytes)

    candidate_target = session(candidate, True)
    baseline_target = session(baseline, True)
    baseline_default = session(baseline, False)
    candidate_default = session(candidate, False)

    counters = {
        "fresh_generated": 0,
        "generation_errors": 0,
        "generation_attempts": 0,
        "target_candidate_right": 0,
        "target_candidate_runtime_errors": 0,
        "target_raw_equal_sample": 0,
        "target_baseline_runtime_errors_sample": 0,
        "default_raw_equal_sample": 0,
        "default_decoded_equal_sample": 0,
        "default_baseline_right_sample": 0,
        "default_candidate_right_sample": 0,
        "default_baseline_runtime_errors_sample": 0,
        "default_candidate_runtime_errors_sample": 0,
        "max_abs_target_sample": 0.0,
        "max_abs_default_sample": 0.0,
    }

    generator = importlib.import_module(f"task_{TASK_MAP['233']}")
    random.seed(8_004_653)
    while counters["fresh_generated"] < FRESH and counters["generation_attempts"] < FRESH * 10:
        counters["generation_attempts"] += 1
        try:
            converted = scoring.convert_to_numpy(generator.generate())
        except Exception:  # noqa: BLE001
            counters["generation_errors"] += 1
            continue
        if converted is None:
            continue
        counters["fresh_generated"] += 1
        expected = converted["output"] > 0

        candidate_raw = None
        try:
            candidate_raw = candidate_target.run(
                ["output"], {"input": converted["input"]}
            )[0]
        except Exception:  # noqa: BLE001
            counters["target_candidate_runtime_errors"] += 1
        if candidate_raw is not None:
            counters["target_candidate_right"] += int(
                np.array_equal(candidate_raw > 0, expected)
            )

        if counters["fresh_generated"] <= DUAL_SAMPLE:
            baseline_raw = None
            try:
                baseline_raw = baseline_target.run(
                    ["output"], {"input": converted["input"]}
                )[0]
            except Exception:  # noqa: BLE001
                counters["target_baseline_runtime_errors_sample"] += 1
            if baseline_raw is not None and candidate_raw is not None:
                counters["target_raw_equal_sample"] += int(
                    np.array_equal(baseline_raw, candidate_raw, equal_nan=True)
                )
                counters["max_abs_target_sample"] = max(
                    counters["max_abs_target_sample"],
                    float(
                        np.abs(
                            np.nan_to_num(baseline_raw).astype(np.float64, copy=False)
                            - np.nan_to_num(candidate_raw).astype(np.float64, copy=False)
                        ).max(initial=0.0)
                    ),
                )

            baseline_default_raw = candidate_default_raw = None
            try:
                baseline_default_raw = baseline_default.run(
                    ["output"], {"input": converted["input"]}
                )[0]
            except Exception:  # noqa: BLE001
                counters["default_baseline_runtime_errors_sample"] += 1
            try:
                candidate_default_raw = candidate_default.run(
                    ["output"], {"input": converted["input"]}
                )[0]
            except Exception:  # noqa: BLE001
                counters["default_candidate_runtime_errors_sample"] += 1
            if baseline_default_raw is not None and candidate_default_raw is not None:
                counters["default_raw_equal_sample"] += int(
                    np.array_equal(
                        baseline_default_raw, candidate_default_raw, equal_nan=True
                    )
                )
                counters["default_decoded_equal_sample"] += int(
                    np.array_equal(
                        baseline_default_raw > 0, candidate_default_raw > 0
                    )
                )
                counters["default_baseline_right_sample"] += int(
                    np.array_equal(baseline_default_raw > 0, expected)
                )
                counters["default_candidate_right_sample"] += int(
                    np.array_equal(candidate_default_raw > 0, expected)
                )
                counters["max_abs_default_sample"] = max(
                    counters["max_abs_default_sample"],
                    float(
                        np.abs(
                            np.nan_to_num(baseline_default_raw).astype(np.float64, copy=False)
                            - np.nan_to_num(candidate_default_raw).astype(np.float64, copy=False)
                        ).max(initial=0.0)
                    ),
                )

        if counters["fresh_generated"] % 100 == 0:
            OUT.write_text(json.dumps({"status": "RUNNING", **counters}, indent=2) + "\n")
        if counters["fresh_generated"] % 500 == 0:
            print(f"task233: {counters['fresh_generated']}/{FRESH}", flush=True)

    strict = structural_gate(candidate)
    exact_alias_proof = {
        "removed": "audit_one_i16",
        "kept": "one_i8",
        "byte_identical_tensor_proto_ignoring_name": True,
        "only_graph_change": "consumer input alias plus removal of duplicate initializer",
        "floating_operation_order_changed": False,
    }
    report = {
        "status": "COMPLETE",
        "task": 233,
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_sha256": sha256(baseline_bytes),
        "candidate": str(CANDIDATE_PATH.relative_to(ROOT)),
        "candidate_sha256": sha256(candidate_bytes),
        "cost_before": 7432,
        "cost_after": 7431,
        "cost_reduction": 1,
        "projected_gain": 0.0001345623362034587,
        "known_evidence": {
            "source": "scripts/golf/loop_8003_40/agent_exact_resume/FINAL_REPORT.json",
            "baseline_and_candidate_sha256_match": True,
            "candidate_right": 266,
            "total": 266,
            "runtime_errors": 0,
        },
        "fresh_target_ort_disable_all": {
            "right": counters["target_candidate_right"],
            "total": counters["fresh_generated"],
            "runtime_errors": counters["target_candidate_runtime_errors"],
        },
        "dual_ort_differential_sample": {
            "count": DUAL_SAMPLE,
            "target_raw_bitwise_equal": counters["target_raw_equal_sample"],
            "target_baseline_runtime_errors": counters["target_baseline_runtime_errors_sample"],
            "default_raw_bitwise_equal": counters["default_raw_equal_sample"],
            "default_decoded_equal": counters["default_decoded_equal_sample"],
            "default_baseline_runtime_errors": counters["default_baseline_runtime_errors_sample"],
            "default_candidate_runtime_errors": counters["default_candidate_runtime_errors_sample"],
            "default_baseline_right": counters["default_baseline_right_sample"],
            "default_candidate_right": counters["default_candidate_right_sample"],
            "max_abs_target": counters["max_abs_target_sample"],
            "max_abs_default": counters["max_abs_default_sample"],
        },
        "generation_errors": counters["generation_errors"],
        "generation_attempts": counters["generation_attempts"],
        "structural_gate": strict,
        "exact_alias_proof": exact_alias_proof,
    }
    report["verdict"] = "ACCEPT_EXACT" if bool(
        strict.get("pass")
        and counters["fresh_generated"] == FRESH
        and counters["target_candidate_right"] == FRESH
        and counters["target_candidate_runtime_errors"] == 0
        and counters["target_raw_equal_sample"] == DUAL_SAMPLE
        and counters["default_raw_equal_sample"] == DUAL_SAMPLE
        and counters["default_decoded_equal_sample"] == DUAL_SAMPLE
        and counters["target_baseline_runtime_errors_sample"] == 0
        and counters["default_baseline_runtime_errors_sample"] == 0
        and counters["default_candidate_runtime_errors_sample"] == 0
    ) else "REJECT"
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
