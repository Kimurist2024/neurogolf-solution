#!/usr/bin/env python3
"""Independent 8012.15 re-audit of the pending task175 POLICY95 lineage."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys

import onnx

import common


SOURCE = common.ROOT / "scripts/golf/cost101_250_half_307/policy95_history_candidates/task175_cost145_40a940588083_POLICY95.onnx"
DEST = common.CANDIDATES / "task175_policy95_cost145.onnx"
OUT = common.HERE / "task175_policy95_rebase8012_audit.json"


def compact(base, row):
    result = base.compact(row)
    result["pass"] = base.row_pass(row)
    return result


def main() -> int:
    common.HERE.mkdir(parents=True, exist_ok=True)
    common.CANDIDATES.mkdir(parents=True, exist_ok=True)
    common.validate_authority()
    base = common.import_path(
        "restart8012_low_policy95_base",
        common.ROOT / "scripts/golf/cost101_250_half_307/scan_policy95_history.py",
    )
    support = base.load_support()
    support.POLICY_THRESHOLD = 0.95
    data = SOURCE.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != "40a9405880836a60f100e0072b476e4383c12c7ee053eb12ada1f049ee2e8d7c":
        raise RuntimeError("candidate SHA mismatch")
    model = onnx.load_model_from_string(data)
    cases, counts = support.known_cases(175)
    profile = base.fast_profile(support, 175, model, cases[0])
    structure = base.structure_audit(support, 175, model, data)
    known = support.evaluate_four(data, cases)
    task_map = json.loads((common.ROOT / "docs/golf/task_hash_map.json").read_text())
    # This lane's helper is also named ``common.py``.  The ARC generator
    # intentionally imports its own tasks/common.py by the bare name
    # ``common``; install that exact module before importing the generator.
    generator_common_path = common.ROOT / "inputs/arc-gen-repo/tasks/common.py"
    generator_spec = importlib.util.spec_from_file_location("common", generator_common_path)
    if generator_spec is None or generator_spec.loader is None:
        raise RuntimeError(generator_common_path)
    generator_common = importlib.util.module_from_spec(generator_spec)
    sys.modules["common"] = generator_common
    generator_spec.loader.exec_module(generator_common)
    fresh = []
    for seed in (405_200_175, 405_300_175):
        generated, generation = support.fresh_cases(175, seed, task_map)
        rows = support.evaluate_four(data, generated)
        fresh.append({
            "seed": seed, "generation": generation,
            "runtime": {name: compact(base, row) for name, row in rows.items()},
            "pass": all(base.row_pass(row) for row in rows.values()),
        })
    known_rows = {name: compact(base, row) for name, row in known.items()}
    authority_cost = common.current_costs()[175]
    candidate_cost = None if profile is None else int(profile["cost"])
    passed = bool(
        authority_cost == 166 and candidate_cost == 145 and structure["pass"]
        and all(base.row_pass(row) for row in known.values())
        and all(run["pass"] for run in fresh)
    )
    if passed:
        DEST.write_bytes(data)
    payload = {
        "authority": {"zip": str(common.AUTHORITY.relative_to(common.ROOT)), "sha256": common.AUTHORITY_SHA256, "lb": 8012.15},
        "authority_diff": common.authority_diff(),
        "task": 175, "source": str(SOURCE.relative_to(common.ROOT)),
        "source_sha256": digest, "saved_candidate": str(DEST.relative_to(common.ROOT)) if passed else None,
        "authority_cost": authority_cost, "candidate_profile": profile,
        "score_gain": None if candidate_cost is None else math.log(authority_cost / candidate_cost),
        "half_target_met": bool(candidate_cost is not None and candidate_cost * 2 <= authority_cost),
        "known_counts": counts, "known_four": known_rows,
        "fresh": fresh, "structure": structure,
        "policy95_threshold": 0.95, "policy95_pass": passed,
        "admission_threshold": 0.90, "policy90_admitted": passed,
        "classification": "ADMITTED_POLICY90_NONBLACK_NOT_LB_GUARANTEED",
        "rejection_gates": "errors/nonfinite/shape/small-positive/config-sign mismatch/UB/cloak all fail closed",
        "protected_writes": "lane only; authority/root/others untouched",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": passed, "authority_cost": authority_cost, "candidate_cost": candidate_cost,
        "known": {name: row["accuracy"] for name, row in known_rows.items()},
        "fresh": [[row["accuracy"] for row in run["runtime"].values()] for run in fresh],
        "score_gain": payload["score_gain"], "out": str(OUT.relative_to(common.ROOT)),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
