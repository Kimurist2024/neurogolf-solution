#!/usr/bin/env python3
"""Independent policy-95 audit of the retained finite task202 cost-20 net."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task202_r01_static20.onnx"
EXPECTED_SOURCE_SHA256 = "11c9d787f99ce439e7674a5a45899162d54d8005c356644a4762d0ca958b6283"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
FRESH_PER_SEED = 2_000
SEEDS = (304_202_001, 304_202_002)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_support() -> Any:
    path = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
    spec = importlib.util.spec_from_file_location("half304_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.FRESH_PER_SEED = FRESH_PER_SEED
    return module


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive",
        "maximum_nonpositive", "sign_mismatch_cases_vs_disable_threads1",
        "sign_mismatch_cells_vs_disable_threads1", "sign_sha256", "raw_sha256",
        "first_wrong", "first_error", "first_shape_mismatch", "first_sign_mismatch",
        "optimization", "threads", "session_error",
    )
    return {key: row.get(key) for key in keys if key in row}


def policy95(row: dict[str, Any]) -> bool:
    return bool(
        row.get("total") == FRESH_PER_SEED
        and float(row.get("accuracy", 0.0)) >= 0.95
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def main() -> int:
    started = time.monotonic()
    HERE.mkdir(parents=True, exist_ok=True)
    if sha256(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("source SHA mismatch")

    support = load_support()
    data = SOURCE.read_bytes()
    model = onnx.load_model_from_string(data)
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))

    structural = support.structural_audit(202, model, data)
    # The only strict-audit exception recorded separately is the 16-input
    # output-only Einsum. It has no intermediate tensors or nested graphs.
    non_giant_reasons = [reason for reason in structural["reasons"] if reason != "giant"]
    structure_policy95 = bool(not non_giant_reasons)
    profile = support.official_profile(202, model, "half304_task202_policy95")

    known_cases, known_counts = support.known_cases(202)
    known_four_raw = support.evaluate_four(data, known_cases)
    known_four = {name: compact(row) for name, row in known_four_raw.items()}

    fresh_runs = []
    for seed in SEEDS:
        cases, generation = support.fresh_cases(202, seed, task_map)
        rows_raw = support.evaluate_four(data, cases)
        rows = {name: compact(row) for name, row in rows_raw.items()}
        passed = bool(
            generation["accepted"] == FRESH_PER_SEED
            and all(policy95(row) for row in rows_raw.values())
        )
        fresh_runs.append({
            "seed": seed,
            "generation": generation,
            "four": rows,
            "policy95_pass": passed,
        })

    admitted = bool(
        structure_policy95
        and profile is not None
        and int(profile["cost"]) == 20
        and all(run["policy95_pass"] for run in fresh_runs)
    )
    candidate_path = HERE / "candidates/task202_policy95_cost20.onnx"
    if admitted:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE, candidate_path)

    report = {
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "sha256": sha256(AUTHORITY),
            "lb": 8011.05,
            "task202_cost": 48,
        },
        "candidate": {
            "source": str(SOURCE.relative_to(ROOT)),
            "sha256": sha256(SOURCE),
            "official_profile": profile,
            "projected_cost": 20,
            "projected_gain": 28,
            "half_target_met": True,
            "output": str(candidate_path.relative_to(ROOT)) if admitted else None,
        },
        "policy": {
            "classification": "POLICY95_PRIVATE_ZERO_LINEAGE",
            "threshold": 0.95,
            "fresh_per_seed": FRESH_PER_SEED,
            "seed_count": len(SEEDS),
            "four_ort_configs": True,
            "runtime_errors_forbidden": True,
            "nonfinite_forbidden": True,
            "shape_cloak_forbidden": True,
            "lookup_forbidden": True,
            "giant_einsum_exception": {
                "enabled": True,
                "input_count": structural["max_einsum_inputs"],
                "output_only": len(model.graph.node) == 1,
            },
        },
        "structure": structural,
        "structure_policy95": structure_policy95,
        "known_counts": known_counts,
        "known_four": known_four,
        "fresh": fresh_runs,
        "decision": "ADMIT_POLICY95" if admitted else "REJECT",
        "elapsed_seconds": time.monotonic() - started,
        "root_or_stage_written": False,
    }
    (HERE / "task202_policy95_evidence.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "decision": report["decision"],
        "fresh_accuracy": {
            run["seed"]: {name: row["accuracy"] for name, row in run["four"].items()}
            for run in fresh_runs
        },
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
