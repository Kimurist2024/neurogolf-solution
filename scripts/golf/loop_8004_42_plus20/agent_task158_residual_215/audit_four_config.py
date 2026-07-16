#!/usr/bin/env python3
"""Known/fresh raw-equivalence audit across ORT mode x thread count."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "others/71407/task158.onnx"
CANDIDATE = HERE / "candidates/task158_exact_anchor_role_bits.onnx"
SHARED = ROOT / "scripts/golf/loop_8004_42_plus20/root_selu_scan_127/audit_candidates.py"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH_SEEDS = (1_582_151, 1_582_152)
FRESH_PER_SEED = 3_000


def load_shared():
    spec = importlib.util.spec_from_file_location("task158_role_bits_shared", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load shared audit helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    shared = load_shared()
    authority = AUTHORITY.read_bytes()
    candidate = CANDIDATE.read_bytes()
    known_cases = shared.known(158)
    report: dict[str, object] = {
        "task": 158,
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "configs": [label for _disable, _threads, label in CONFIGS],
        "known_four_configs": {},
        "fresh": [],
    }
    for disable, threads, label in CONFIGS:
        report["known_four_configs"][label] = shared.evaluate_cases(
            authority, candidate, known_cases, disable, threads
        )
        print("known", label, flush=True)
    for seed in FRESH_SEEDS:
        cases, attempts = shared.generate(158, seed, FRESH_PER_SEED)
        stream = {"seed": seed, "attempts": attempts, "modes": {}}
        for disable, threads, label in CONFIGS:
            stream["modes"][label] = shared.evaluate_cases(
                authority, candidate, cases, disable, threads
            )
            print("fresh", seed, label, flush=True)
        report["fresh"].append(stream)
    comparisons = list(report["known_four_configs"].values()) + [
        row
        for stream in report["fresh"]
        for row in stream["modes"].values()
    ]
    report["summary"] = {
        "known_cases_per_config": len(known_cases),
        "known_raw_comparisons": len(known_cases) * len(CONFIGS),
        "fresh_cases_per_seed_per_config": FRESH_PER_SEED,
        "fresh_raw_comparisons": (
            len(FRESH_SEEDS) * FRESH_PER_SEED * len(CONFIGS)
        ),
        "all_raw_equivalent": all(row.get("exact_equivalent") for row in comparisons),
        "all_truth_correct": all(row.get("perfect_truth") for row in comparisons),
        "runtime_errors_total": sum(
            row.get("runtime_errors_total", 0) for row in comparisons
        ),
        "candidate_nonfinite_total": sum(
            row.get("nonfinite_values", {}).get("candidate", 0)
            for row in comparisons
        ),
        "authority_nonfinite_total": sum(
            row.get("nonfinite_values", {}).get("baseline", 0)
            for row in comparisons
        ),
    }
    summary = report["summary"]
    report["pass"] = bool(
        summary["all_raw_equivalent"]
        and summary["all_truth_correct"]
        and summary["runtime_errors_total"] == 0
        and summary["candidate_nonfinite_total"] == 0
        and summary["authority_nonfinite_total"] == 0
    )
    output = HERE / "evidence/four_config_raw_equivalence.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], **summary}, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
