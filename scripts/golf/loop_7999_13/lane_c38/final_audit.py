#!/usr/bin/env python3
"""Finalize the strict exact-only task398 audit for lane C38."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = HERE / "baseline" / "task398.onnx"
C11_AUDITOR = HERE.parent / "lane_c11" / "audit_candidates.py"
A34 = HERE.parent / "lane_a34"
C13 = HERE.parent / "lane_c13"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_auditor():
    spec = importlib.util.spec_from_file_location("c38_c11_auditor", C11_AUDITOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(C11_AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def known_total(row: dict[str, Any], key: str) -> dict[str, int]:
    return row[key]["total"]


def main() -> int:
    ort.set_default_logger_severity(4)
    exact = read_json(HERE / "exact_structure_audit.json")
    baseline_external = read_json(HERE / "task398_baseline_external.json")
    q4_external = read_json(A34 / "external_validator_task398_allow_mismatch.json")
    archive = read_json(C13 / "candidate_audit.json")

    auditor = load_auditor()
    baseline = auditor.audit("lane_c38_baseline_task398", 398, BASELINE)
    disabled = known_total(baseline, "known_disable_all")
    default = known_total(baseline, "known_default")
    shape_trace = baseline["runtime_shape_trace"]
    score = baseline["official_like_score"]

    structure_pass = all(
        (
            baseline["full_check"],
            baseline["strict_shape_data_prop"],
            not baseline["nonstandard_domains"],
            not baseline["banned_ops"],
            baseline["nested_graph_attributes"] == 0,
            baseline["function_count"] == 0,
            baseline["sparse_initializer_count"] == 0,
            not baseline["conv_bias_findings"],
            not shape_trace["declared_actual_mismatches"],
            shape_trace["undeclared_intermediate_count"] == 0,
            shape_trace["single_example_intermediate_bytes"] == score["memory"] == 144,
        )
    )
    known_pass = all(
        row == {"right": 268, "wrong": 0, "errors": 0}
        for row in (disabled, default)
    )

    archive_rejections = []
    for label in ("task398_r01", "task398_r02", "task398_r03"):
        row = archive[label]
        archive_rejections.append(
            {
                "label": label,
                "sha256": row["sha256"],
                "cost": row["official_like_score"]["cost"],
                "disable_all": row["known_disable_all"]["total"],
                "default": row["known_default"]["total"],
                "reason": "wrong on every known case",
            }
        )

    q4_diff = q4_external["differential"]
    q4_rejection = {
        "label": "task398_q4_d_reuse_347",
        "sha256": q4_external["candidate"]["sha256"],
        "cost": q4_external["candidate"]["cost"],
        "known": q4_external["candidate"]["known"],
        "random_seed": 80004604,
        "random_requested": q4_diff["requested"],
        "random_executable": q4_diff["executable"],
        "threshold_mismatches": q4_diff["mismatches"],
        "first_mismatch": q4_diff["first_mismatch"],
        "reason": (
            "The Q4-to-D gauge is not arbitrary-input exact. Four of 500 independent "
            "seeded cases cross the output threshold."
        ),
    }

    output = {
        "task": 398,
        "scope": "arbitrary-input meaning-preserving exact reductions only",
        "baseline": baseline,
        "baseline_external_validator": baseline_external,
        "baseline_gate": {
            "authoritative_sha256": "741d07c3cd4fa9cfe363aeb30573cb97edda0881857abeb5ae096b77773018e4",
            "sha_matches": baseline["sha256"]
            == "741d07c3cd4fa9cfe363aeb30573cb97edda0881857abeb5ae096b77773018e4",
            "actual_cost": score,
            "structure_pass": structure_pass,
            "known_dual_pass": known_pass,
            "external_known_pass": baseline_external["candidate"]["known"]
            == {
                "right": 268,
                "wrong": 0,
                "skipped": 0,
                "errors": 0,
                "total_seen": 268,
            },
            "existing_giant_einsum_inputs": 69,
            "policy": "grandfather baseline only; no new or enlarged giant Einsum",
        },
        "exact_structure_audit": exact,
        "rejected": {
            "q4_carrier": q4_rejection,
            "archived_cost332": archive_rejections,
        },
        "candidate_gates": {
            "cheaper_exact_candidate_found": False,
            "fresh_dual_5000": {
                "run": False,
                "reason": "No cheaper arbitrary-input exact candidate reached the fresh gate.",
            },
            "external_seed80004604_cases500": {
                "run": False,
                "reason": "No cheaper arbitrary-input exact candidate reached the external gate.",
            },
        },
        "decision": {
            "winner": None,
            "cost_reduction": 0,
            "projected_score_gain": 0.0,
            "baseline_preserved": True,
            "reason": (
                "No exact score-bearing duplicate, identity, carrier, axis, or factor reduction "
                "survives the dense-only and no-enlarged-giant constraints."
            ),
        },
    }
    output_path = HERE / "final_audit.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "task": 398,
                "sha256": baseline["sha256"],
                "cost": score["cost"],
                "structure_pass": structure_pass,
                "known_dual_pass": known_pass,
                "winner": None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
