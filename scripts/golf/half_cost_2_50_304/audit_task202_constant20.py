#!/usr/bin/env python3
"""Build and audit a color-symmetric task202 cost-20 policy-95 candidate."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

import audit_task202_policy95 as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task202_r01_static20.onnx"
SEED = 304_202_205
FOREGROUND_CODE = np.float32(0.1005)


def build() -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    values = np.asarray([-1.0] + [float(FOREGROUND_CODE)] * 9, dtype=np.float32)
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name in {"VH", "VV"}:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(values, initializer.name)
            )
    model.producer_name = "half_cost_2_50_304_constant20"
    return model


def main() -> int:
    started = time.monotonic()
    if base.sha256(base.AUTHORITY) != base.AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    support = base.load_support()
    model = build()
    data = model.SerializeToString()
    attempt = HERE / "attempts/task202_constant01005_cost20.onnx"
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_bytes(data)
    structure = support.structural_audit(202, model, data)
    non_giant_reasons = [reason for reason in structure["reasons"] if reason != "giant"]
    profile = support.official_profile(202, model, "half304_task202_constant20")
    known_cases, known_counts = support.known_cases(202)
    known_raw = support.evaluate_four(data, known_cases)
    known = {name: base.compact(row) for name, row in known_raw.items()}
    task_map = json.loads(base.TASK_MAP.read_text(encoding="utf-8"))
    fresh_cases, generation = support.fresh_cases(202, SEED, task_map)
    fresh_raw = support.evaluate_four(data, fresh_cases)
    fresh = {name: base.compact(row) for name, row in fresh_raw.items()}
    known_exact = all(
        row.get("right") == row.get("total")
        and row.get("errors") == 0 and row.get("nonfinite_cases") == 0
        and row.get("runtime_shape_mismatches") == 0
        for row in known_raw.values()
    )
    admitted = bool(
        not non_giant_reasons and known_exact
        and profile is not None and int(profile["cost"]) == 20
        and all(base.policy95(row) for row in fresh_raw.values())
    )
    output = HERE / "candidates/task202_policy95_constant01005_cost20.onnx"
    if admitted:
        output.write_bytes(data)
    report = {
        "authority": {
            "path": str(base.AUTHORITY.relative_to(ROOT)),
            "sha256": base.sha256(base.AUTHORITY),
            "lb": 8011.05,
            "task202_cost": 48,
        },
        "candidate": {
            "source_template": str(SOURCE.relative_to(ROOT)),
            "sha256": base.sha256(attempt),
            "foreground_code": float(FOREGROUND_CODE),
            "official_profile": profile,
            "projected_cost": 20,
            "projected_gain": 28,
            "half_target_met": True,
            "output": str(output.relative_to(ROOT)) if admitted else None,
        },
        "policy": {
            "classification": "POLICY95_NEW_CONSTANT_VARIANT",
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
        "known_exact_all_configs": known_exact,
        "fresh": {"seed": SEED, "generation": generation, "four": fresh},
        "decision": "ADMIT_POLICY95" if admitted else "REJECT",
        "elapsed_seconds": time.monotonic() - started,
        "root_or_stage_written": False,
    }
    (HERE / "task202_constant20_evidence.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "decision": report["decision"],
        "known_exact": known_exact,
        "fresh_accuracy": {name: row["accuracy"] for name, row in fresh.items()},
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
