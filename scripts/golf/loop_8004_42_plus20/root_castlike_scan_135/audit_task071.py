#!/usr/bin/env python3
"""Dual-runtime raw-equivalence audit for the sole strict-lower Cast winner."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = Path("/private/tmp/ng800946_rank/task071.onnx")
CANDIDATE = HERE / "candidates/task071.onnx"
SELU_LANE = ROOT / "scripts/golf/loop_8004_42_plus20/root_selu_scan_127"
sys.path.insert(0, str(SELU_LANE))

from audit_candidates import evaluate_cases, generate, known, runtime_shape_truth  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    baseline = AUTHORITY.read_bytes()
    candidate = CANDIDATE.read_bytes()
    known_cases = known(71)
    known_rows = {}
    for disable, threads, label in (
        (True, 1, "disable_all_threads1"),
        (True, 4, "disable_all_threads4"),
        (False, 1, "default_threads1"),
        (False, 4, "default_threads4"),
    ):
        print(f"known {label}", flush=True)
        known_rows[label] = evaluate_cases(
            baseline, candidate, known_cases, disable=disable, threads=threads
        )
    fresh_rows = {}
    for seed in (135_071_001, 135_071_002):
        cases, attempts = generate(71, seed, 1500)
        for disable, label in ((True, "disable_all"), (False, "default")):
            key = f"seed{seed}_{label}"
            print(f"fresh {key}", flush=True)
            fresh_rows[key] = evaluate_cases(
                baseline, candidate, cases, disable=disable, threads=1
            )
            fresh_rows[key]["attempts"] = attempts
    all_rows = list(known_rows.values()) + list(fresh_rows.values())
    result = {
        "task": 71,
        "semantic_identity": "CastLike(x, fixed int32 witness) == Cast(x, to=INT32)",
        "authority_sha256": digest(AUTHORITY),
        "candidate_sha256": digest(CANDIDATE),
        "known": known_rows,
        "fresh": fresh_rows,
        "authority_runtime_shape": runtime_shape_truth(71, baseline),
        "candidate_runtime_shape": runtime_shape_truth(71, candidate),
        "pass": all(
            row.get("exact_equivalent")
            and row.get("runtime_errors_total") == 0
            and row.get("candidate_accuracy", 0.0) >= 0.90
            for row in all_rows
        ),
    }
    (HERE / "audit_task071.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "pass": result["pass"],
        "known_min_accuracy": min(row["candidate_accuracy"] for row in known_rows.values()),
        "fresh_min_accuracy": min(row["candidate_accuracy"] for row in fresh_rows.values()),
        "raw_equal": all(row["exact_equivalent"] for row in all_rows),
    }, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
