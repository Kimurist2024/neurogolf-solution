#!/usr/bin/env python3
"""Independent POLICY95 audit of the known-LB-zero task343 cost172 model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "scripts/golf/scratch_codex_7994/task343_sound/artifact_black172.onnx"
DEST = HERE / "policy95_candidates/task343_cost172_POLICY95_KNOWN_LB_ZERO.onnx"


def load_support():
    path = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
    spec = importlib.util.spec_from_file_location("audit343_307_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compact(row):
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive", "maximum_nonpositive",
        "sign_mismatch_cases_vs_disable_threads1", "sign_mismatch_cells_vs_disable_threads1",
        "sign_sha256", "raw_sha256", "first_wrong", "first_error", "optimization", "threads",
    )
    return {key: row.get(key) for key in keys if key in row}


def runtime_pass(row, threshold, *, exact=False):
    accuracy = float(row.get("accuracy", 0.0))
    return bool(
        accuracy == 1.0 if exact else accuracy >= threshold
    ) and bool(
        row.get("errors") == 0 and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0 and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def main():
    support = load_support()
    support.POLICY_THRESHOLD = 0.95
    data = SOURCE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = "c1047d40b875d37a7a9e28a52a47e2c569f5156924691118082aaca4ed5198e6"
    if digest != expected:
        raise RuntimeError(f"source SHA mismatch {digest}")
    model = onnx.load_model_from_string(data)
    structure = support.structural_audit(343, model, data)
    profile = support.official_profile(343, model, "policy95_307")
    known_cases, known_generation = support.known_cases(343)
    known = support.evaluate_four(data, known_cases)
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    fresh_runs = []
    for seed in (307_000_343, 307_100_343):
        cases, generation = support.fresh_cases(343, seed, task_map)
        rows = support.evaluate_four(data, cases)
        fresh_runs.append({
            "seed": seed, "generation": generation,
            "runtime": {name: compact(row) for name, row in rows.items()},
            "pass_policy95": all(runtime_pass(row, 0.95) for row in rows.values()),
        })
    known_pass = all(runtime_pass(row, 1.0, exact=True) for row in known.values())
    policy95_pass = bool(
        structure["pass"] and profile and int(profile["cost"]) == 172
        and known_pass and all(row["pass_policy95"] for row in fresh_runs)
    )
    if policy95_pass:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        DEST.write_bytes(data)
    authority_score = 25.0 - math.log(173)
    candidate_score = 25.0 - math.log(172)
    payload = {
        "task": 343, "classification": (
            "PASS_POLICY95_KNOWN_LB_ZERO_NOT_GUARANTEED" if policy95_pass else "REJECT"
        ),
        "known_lb_history": {
            "private_zero_at_cost172": True,
            "evidence": "memory/current-best.md: task343@172 individually LB-zero",
        },
        "authority": {"cost": 173, "task_score": authority_score},
        "candidate": {
            "source": str(SOURCE.relative_to(ROOT)), "saved_path": str(DEST.relative_to(ROOT)),
            "sha256": digest, "profile": profile, "task_score": candidate_score,
            "score_gain": candidate_score - authority_score, "half_target": 86,
            "meets_half_target": False,
        },
        "structure": structure,
        "known_generation": known_generation,
        "known_runtime": {name: compact(row) for name, row in known.items()},
        "known_exact_four": known_pass,
        "fresh": {"threshold": 0.95, "cases_per_seed": support.FRESH_PER_SEED,
                  "runs": fresh_runs, "pass": all(row["pass_policy95"] for row in fresh_runs)},
        "admission": policy95_pass,
    }
    (HERE / "task343_policy95_evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "classification": payload["classification"], "profile": profile,
        "known": {name: row["accuracy"] for name, row in known.items()},
        "fresh": [{"seed": row["seed"], "pass": row["pass_policy95"],
                   "accuracy": {name: value["accuracy"] for name, value in row["runtime"].items()}}
                  for row in fresh_runs],
    }, indent=2))


if __name__ == "__main__":
    main()
