#!/usr/bin/env python3
"""Consolidate the 16-candidate, threads-1/4 task198 fresh audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HIGH47 = ROOT / "scripts/golf/loop_8004_42_plus20/agent_high47"


def main() -> None:
    rows = []
    for model in sorted((HIGH47 / "candidates").glob("task198*.onnx")):
        stem = model.stem
        sources = {
            "1": Path("/tmp") / f"{stem}_fresh1000.json",
            "4": Path("/tmp") / f"{stem}_fresh1000_t4.json",
        }
        data = {threads: json.loads(path.read_text()) for threads, path in sources.items()}
        configs = {}
        for threads, document in data.items():
            configs[f"threads{threads}"] = []
            for fresh in document["fresh"]:
                configs[f"threads{threads}"].append(
                    {
                        "seed": fresh["seed"],
                        "reference": fresh["reference"],
                        "disable_all": fresh["model"]["disable_all"],
                        "default": fresh["model"]["default"],
                        "disable_default_correctness_disagreements": fresh[
                            "mode_result_disagreements"
                        ],
                    }
                )
        t1_right = sum(x["disable_all"]["right"] for x in configs["threads1"])
        t4_right = sum(x["disable_all"]["right"] for x in configs["threads4"])
        total = sum(x["disable_all"]["total"] for x in configs["threads1"])
        rows.append(
            {
                "model": str(model.relative_to(ROOT)),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "total_each_thread_mode": total,
                "threads1_right": t1_right,
                "threads4_right": t4_right,
                "threads1_rate": t1_right / total,
                "threads4_rate": t4_right / total,
                "threads1_threads4_right_counts_equal": t1_right == t4_right,
                "any_runtime_errors": any(
                    config[mode]["errors"]
                    for configs_for_threads in configs.values()
                    for config in configs_for_threads
                    for mode in ("disable_all", "default")
                ),
                "any_near_margin": any(
                    config[mode]["near_margin_count"]
                    for configs_for_threads in configs.values()
                    for config in configs_for_threads
                    for mode in ("disable_all", "default")
                ),
                "configs": configs,
            }
        )

    best = max(rows, key=lambda row: row["threads1_right"])
    result = {
        "task": 198,
        "generator_hash": "83302e8f",
        "candidate_count": len(rows),
        "fresh_examples_per_seed": 1000,
        "seeds": [47_000_199, 47_100_199],
        "reference_correct_per_seed": [1000, 1000],
        "reference_total": 2000,
        "all_thread_and_optimization_runs_runtime_error_free": not any(
            row["any_runtime_errors"] for row in rows
        ),
        "all_thread_and_optimization_runs_near_margin_free": not any(
            row["any_near_margin"] for row in rows
        ),
        "all_candidates_have_identical_threads1_threads4_right_counts": all(
            row["threads1_threads4_right_counts_equal"] for row in rows
        ),
        "candidates_with_100_percent_generator_accuracy": [],
        "best_empirical_candidate": {
            "model": best["model"],
            "right": best["threads1_right"],
            "total": best["total_each_thread_mode"],
            "rate": best["threads1_rate"],
            "verdict": "reject: legal generator counterexamples exist",
        },
        "rows": rows,
    }
    out = HERE / "fresh_matrix.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}: {len(rows)} candidates")


if __name__ == "__main__":
    main()
