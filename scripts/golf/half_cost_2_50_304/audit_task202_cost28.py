#!/usr/bin/env python3
"""Independent four-ORT/fresh audit of the retained task202 cost-28 fallback."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import onnx

import audit_task202_policy95 as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task202_r03_static28.onnx"
EXPECTED_SHA256 = "cd442a7cb2df00311044258ded1c9e1635bb19193fba6940e4b803831d6d03ea"
SEED = 304_202_028


def main() -> int:
    started = time.monotonic()
    if base.sha256(base.AUTHORITY) != base.AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("source SHA mismatch")
    support = base.load_support()
    data = SOURCE.read_bytes()
    model = onnx.load_model_from_string(data)
    task_map = json.loads(base.TASK_MAP.read_text(encoding="utf-8"))
    structure = support.structural_audit(202, model, data)
    non_giant_reasons = [reason for reason in structure["reasons"] if reason != "giant"]
    profile = support.official_profile(202, model, "half304_task202_cost28")
    known_cases, known_counts = support.known_cases(202)
    known_raw = support.evaluate_four(data, known_cases)
    known = {name: base.compact(row) for name, row in known_raw.items()}
    fresh_cases, generation = support.fresh_cases(202, SEED, task_map)
    fresh_raw = support.evaluate_four(data, fresh_cases)
    fresh = {name: base.compact(row) for name, row in fresh_raw.items()}
    admitted = bool(
        not non_giant_reasons
        and profile is not None and int(profile["cost"]) == 28
        and all(base.policy95(row) for row in fresh_raw.values())
    )
    exact_fresh = bool(
        admitted and all(row.get("right") == row.get("total") for row in fresh_raw.values())
    )
    output = HERE / "candidates/task202_fallback_cost28.onnx"
    if admitted:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE, output)
    report = {
        "authority": {
            "path": str(base.AUTHORITY.relative_to(ROOT)),
            "sha256": base.sha256(base.AUTHORITY),
            "lb": 8011.05,
            "task202_cost": 48,
        },
        "candidate": {
            "source": str(SOURCE.relative_to(ROOT)),
            "sha256": EXPECTED_SHA256,
            "official_profile": profile,
            "projected_cost": 28,
            "projected_gain": 20,
            "half_target_met": False,
            "output": str(output.relative_to(ROOT)) if admitted else None,
        },
        "policy": {
            "classification": "POLICY95_PRIVATE_ZERO_LINEAGE_FALLBACK",
            "threshold": 0.95,
            "fresh_per_seed": base.FRESH_PER_SEED,
            "four_ort_configs": True,
            "giant_einsum_exception": {
                "enabled": True,
                "input_count": structure["max_einsum_inputs"],
                "output_only": len(model.graph.node) == 1,
            },
        },
        "structure": structure,
        "known_counts": known_counts,
        "known_four": known,
        "fresh": {"seed": SEED, "generation": generation, "four": fresh},
        "fresh_exact_all_configs": exact_fresh,
        "decision": "ADMIT_POLICY95" if admitted else "REJECT",
        "elapsed_seconds": time.monotonic() - started,
        "root_or_stage_written": False,
    }
    (HERE / "task202_cost28_evidence.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "decision": report["decision"],
        "fresh_exact_all_configs": exact_fresh,
        "fresh_accuracy": {name: row["accuracy"] for name, row in fresh.items()},
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
