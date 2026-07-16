#!/usr/bin/env python3
"""Build and audit the non-giant task202 cost-20 policy-95 candidate."""

from __future__ import annotations

import json
import time
from pathlib import Path

import onnx
from onnx import helper

import audit_task202_policy95 as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task202_r01_static20.onnx"
SEEDS = (304_202_014, 304_202_114)
REMOVED_ORIGINAL_OPERANDS = (3, 7)


def build() -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    node = model.graph.node[0]
    attribute = next(item for item in node.attribute if item.name == "equation")
    equation = helper.get_attribute_value(attribute).decode()
    lhs, rhs = equation.split("->", 1)
    terms = lhs.split(",")
    names = list(node.input)
    kept_terms = [term for index, term in enumerate(terms) if index not in REMOVED_ORIGINAL_OPERANDS]
    kept_names = [name for index, name in enumerate(names) if index not in REMOVED_ORIGINAL_OPERANDS]
    del node.input[:]
    node.input.extend(kept_names)
    attribute.s = (",".join(kept_terms) + "->" + rhs).encode()
    model.producer_name = "half_cost_2_50_304_arity14"
    return model


def exact(row: dict) -> bool:
    return bool(
        row.get("right") == row.get("total")
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
    if base.sha256(base.AUTHORITY) != base.AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    support = base.load_support()
    model = build()
    data = model.SerializeToString()
    attempt = HERE / "attempts/task202_arity14_cost20.onnx"
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_bytes(data)
    structure = support.structural_audit(202, model, data)
    profile = support.official_profile(202, model, "half304_task202_arity14")
    known_cases, known_counts = support.known_cases(202)
    known_raw = support.evaluate_four(data, known_cases)
    known = {name: base.compact(row) for name, row in known_raw.items()}
    known_exact = all(exact(row) for row in known_raw.values())
    task_map = json.loads(base.TASK_MAP.read_text(encoding="utf-8"))
    fresh_runs = []
    fresh_raw_runs = []
    for seed in SEEDS:
        fresh_cases, generation = support.fresh_cases(202, seed, task_map)
        fresh_raw = support.evaluate_four(data, fresh_cases)
        fresh_raw_runs.append(fresh_raw)
        fresh_runs.append({
            "seed": seed,
            "generation": generation,
            "four": {name: base.compact(row) for name, row in fresh_raw.items()},
        })
    admitted = bool(
        structure["pass"] and known_exact
        and profile is not None and int(profile["cost"]) == 20
        and all(
            base.policy95(row)
            for fresh_raw in fresh_raw_runs
            for row in fresh_raw.values()
        )
    )
    output = HERE / "candidates/task202_policy95_arity14_cost20.onnx"
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
            "official_profile": profile,
            "projected_cost": 20,
            "projected_gain": 28,
            "projected_score_gain": 0.8754687373538999,
            "half_target_met": True,
            "einsum_inputs_before": 16,
            "einsum_inputs_after": 14,
            "removed_original_operands": list(REMOVED_ORIGINAL_OPERANDS),
            "output": str(output.relative_to(ROOT)) if admitted else None,
        },
        "equivalence_argument": {
            "operand_3": "nqse sums the one-hot grid over q,e, yielding the positive active width on every s selected by the neighboring nonzero factors",
            "operand_7": "nvit sums the one-hot grid over v,i, yielding the positive active height on every t selected by the neighboring nonzero factors",
            "sign_effect": "removing both divides every relevant raw output by positive width*height; zeros remain zeros, so threshold(raw>0) is unchanged",
        },
        "policy": {
            "classification": "POLICY95_NON_GIANT",
            "threshold": 0.95,
            "fresh_per_seed": base.FRESH_PER_SEED,
            "fresh_seed_count": len(SEEDS),
            "four_ort_configs": True,
            "giant_einsum_exception": False,
            "runtime_errors_forbidden": True,
            "nonfinite_forbidden": True,
            "shape_cloak_forbidden": True,
            "lookup_forbidden": True,
        },
        "structure": structure,
        "known_counts": known_counts,
        "known_four": known,
        "known_exact_all_configs": known_exact,
        "fresh": fresh_runs,
        "decision": "ADMIT_POLICY95" if admitted else "REJECT",
        "elapsed_seconds": time.monotonic() - started,
        "root_or_stage_written": False,
    }
    (HERE / "task202_arity14_evidence.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "decision": report["decision"],
        "structure_pass": structure["pass"],
        "structure_reasons": structure["reasons"],
        "known_exact": known_exact,
        "fresh_accuracy": {
            run["seed"]: {name: row["accuracy"] for name, row in run["four"].items()}
            for run in fresh_runs
        },
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
