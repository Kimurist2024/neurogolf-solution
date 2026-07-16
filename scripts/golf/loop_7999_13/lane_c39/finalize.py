#!/usr/bin/env python3
"""Aggregate C39 evidence and separate 95%-eligibility from strict recommendation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SEEDS = (80004605, 80004606, 80004607)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    structure = read_json(HERE / "structural_audit.json")
    fresh = read_json(HERE / "fresh_dual_5000.json")
    external = [
        read_json(HERE / f"external_seed{seed}_500.json") for seed in SEEDS
    ]
    base = structure["models"]["baseline"]
    cand = structure["models"]["candidate"]
    fresh_base = fresh["results"]["baseline"]
    fresh_cand = fresh["results"]["candidate"]

    known_dual_pass = all(
        cand[key]["total"] == {"right": 266, "wrong": 0, "errors": 0}
        for key in ("known_disable_all", "known_default")
    )
    fresh_user_gate = all(
        fresh_cand[mode]["accuracy"] >= 0.95
        and fresh_cand[mode]["runtime_errors"] == 0
        for mode in ("disable_all", "default")
    )
    fresh_non_regressing = all(
        fresh_cand[mode]["right"] >= fresh_base[mode]["right"]
        for mode in ("disable_all", "default")
    )
    external_preflight_pass = all(
        row["candidate"]["valid"]
        and row["candidate"]["preflight_ok"]
        and row["candidate"]["known"]["errors"] == 0
        and row["differential"]["skipped_both_failed"] == 0
        and row["differential"]["skipped_one_failed"] == 0
        for row in external
    )
    external_summary = {
        "seeds": list(SEEDS),
        "requested": sum(row["differential"]["requested"] for row in external),
        "executable": sum(row["differential"]["executable"] for row in external),
        "threshold_equal": sum(
            row["differential"]["threshold_equal"] for row in external
        ),
        "threshold_mismatches_vs_baseline": sum(
            row["differential"]["mismatches"] for row in external
        ),
        "skipped_both_failed": sum(
            row["differential"]["skipped_both_failed"] for row in external
        ),
        "skipped_one_failed": sum(
            row["differential"]["skipped_one_failed"] for row in external
        ),
        "preflight_and_runtime_pass": external_preflight_pass,
        "interpretation": (
            "These are arbitrary-grid differential probes, not task-generator gold. Large "
            "baseline/candidate disagreement demonstrates non-equivalence but does not replace "
            "the generator-gold accuracy measurement."
        ),
    }

    cheaper = cand["official_like_score"]["cost"] < base["official_like_score"]["cost"]
    safe_structure = cand["safe_structure"]["pass"]
    adoption_eligible = all(
        (cheaper, safe_structure, known_dual_pass, fresh_user_gate, external_preflight_pass)
    )
    recommended_under_user_policy = adoption_eligible and fresh_non_regressing
    strict_sound_recommended = adoption_eligible and all(
        fresh_cand[mode]["wrong"] == 0 for mode in ("disable_all", "default")
    )

    output = {
        "task": 343,
        "policy": "user permits adoption at >=95% generator-gold accuracy with runtime errors 0",
        "baseline": {
            "sha256": base["sha256"],
            "cost": base["official_like_score"]["cost"],
            "fresh": fresh_base,
        },
        "candidate": {
            "sha256": cand["sha256"],
            "cost": cand["official_like_score"]["cost"],
            "cost_reduction": structure["comparison"]["cost_reduction"],
            "projected_score_gain": structure["comparison"]["projected_score_gain"],
            "known_disable_all": cand["known_disable_all"]["total"],
            "known_default": cand["known_default"]["total"],
            "fresh": fresh_cand,
            "safe_structure": cand["safe_structure"],
        },
        "fresh_comparison": {
            "seed": fresh["seed"],
            "generated": fresh["generated"],
            "generation_errors": fresh["generation_errors"],
            "disable_all": fresh["comparison"]["disable_all"],
            "default": fresh["comparison"]["default"],
            "accuracy_delta_candidate_minus_baseline": {
                mode: fresh_cand[mode]["accuracy"] - fresh_base[mode]["accuracy"]
                for mode in ("disable_all", "default")
            },
            "ort_mode_parity": fresh["ort_mode_parity"],
        },
        "external_validator": external_summary,
        "decision": {
            "adoption_eligible_under_user_95_policy": adoption_eligible,
            "recommended_under_user_95_policy": recommended_under_user_policy,
            "strict_sound_recommended": strict_sound_recommended,
            "candidate_accuracy_non_regressing_on_matched_fresh_corpus": fresh_non_regressing,
            "reason": (
                "Candidate is cheaper, structurally safe, known-complete, runtime-error-free, "
                "and 99.5% accurate on the matched 5000-case generator corpus. It also exceeds "
                "the baseline's 99.26% on that corpus. It remains an approximate classifier "
                "with 25 wrong cases, so it is not a strict 100%-sound promotion."
            ),
        },
    }
    (HERE / "final_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    winner = {
        "task": 343,
        "source": "scripts/golf/loop_7999_13/lane_c39/candidate/task343.onnx",
        "sha256": cand["sha256"],
        "baseline_cost": base["official_like_score"]["cost"],
        "candidate_cost": cand["official_like_score"]["cost"],
        "projected_gain": structure["comparison"]["projected_score_gain"],
        "adoption_basis": "user_95_percent_policy",
        "adoption_eligible": adoption_eligible,
        "recommended_under_user_policy": recommended_under_user_policy,
        "strict_sound": strict_sound_recommended,
        "fresh_accuracy": fresh_cand["disable_all"]["accuracy"],
        "fresh_wrong": fresh_cand["disable_all"]["wrong"],
        "runtime_errors": fresh_cand["disable_all"]["runtime_errors"],
    }
    (HERE / "winner_manifest.json").write_text(
        json.dumps(winner, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
