#!/usr/bin/env python3
"""POLICY95 screen for every cost-251..500 historical strict reduction.

This lane intentionally reuses the already reviewed all-history scanner used
for the 101..250 band.  Only its authority-independent paths are rebound: the
candidate inventory and every output remain in this lane.  The screen is
fail-closed for checker/shape/runtime/bias/margin hazards, while allowing a
known private-zero/lookup formula to be classified as POLICY95 rather than
guaranteed safe when it clears the user's explicit 95% rule.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "scripts/golf/cost101_250_half_307/scan_policy95_history.py"
INVENTORY = ROOT / "scripts/golf/agent_cost251_500_half_307/strict_inventory.json"
PRIVATE_ZERO = {
    9, 15, 18, 35, 44, 48, 49, 66, 70, 72, 77, 86, 90, 93, 96,
    101, 102, 112, 118, 133, 134, 138, 145, 157, 158, 168, 169,
    170, 173, 174, 178, 185, 187, 192, 196, 198, 202, 205, 208,
    209, 216, 219, 222, 233, 246, 251, 255, 273, 277, 285, 286,
    302, 319, 325, 333, 343, 346, 361, 365, 366, 372, 377, 379,
    391, 393, 396,
}


def load_source():
    spec = importlib.util.spec_from_file_location("cost251_500_policy95_402_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if inventory.get("authority_sha256") != (
        "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
    ):
        raise RuntimeError("inventory authority SHA mismatch")
    rows = inventory.get("results", [])
    if not rows or any(not (251 <= int(row["authority_cost"]) <= 500) for row in rows):
        raise RuntimeError("inventory scope mismatch")

    impl = load_source()
    impl.HERE = HERE
    impl.ROOT = ROOT
    impl.THRESHOLD = 0.95
    support = impl.load_support()
    support.POLICY_THRESHOLD = 0.95

    # Profile first.  Most old files have a tiny parameter lower bound but a
    # runtime tensor footprint above today's incumbent; running hundreds of
    # known examples for those cannot affect admission and is needlessly slow.
    started = time.monotonic()
    records = [row for row in rows if not row.get("structural_reasons")]
    known_cache = {}
    screened = []
    eligible = []
    for index, source_row in enumerate(records, 1):
        row = dict(source_row)
        row["stage"] = "load"
        try:
            data = impl.resolve(str(row["source"]))
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise RuntimeError("sha mismatch")
            model = onnx.load_model_from_string(data)
            first_cases, counts = known_cache.get(int(row["task"]), (None, None))
            if first_cases is None:
                first_cases, counts = support.known_cases(int(row["task"]))
                known_cache[int(row["task"])] = (first_cases, counts)
            row["known_counts"] = counts
            profile = impl.fast_profile(support, int(row["task"]), model, first_cases[0])
            row["profile"] = profile
        except Exception as exc:
            row["reject"] = f"load_or_profile:{type(exc).__name__}:{exc}"
            screened.append(row)
            continue
        if profile is None or int(profile["cost"]) >= int(row["authority_cost"]):
            row["stage"] = "profiled"
            row["reject"] = "not_actual_strict_lower"
            screened.append(row)
            continue

        structure = impl.structure_audit(support, int(row["task"]), model, data)
        row["structure"] = structure
        if not structure["pass"]:
            row["stage"] = "structure"
            row["reject"] = "structure:" + ",".join(structure["reasons"])
            screened.append(row)
            continue
        try:
            runtime = support.make_session(data, True, 1)
            known_row, _ = support.evaluate_config(runtime, first_cases, None)
            row["known_disable_threads1"] = impl.compact(known_row)
        except Exception as exc:
            known_row = {"accuracy": 0.0, "errors": 1,
                         "session_error": f"{type(exc).__name__}:{exc}"}
            row["known_disable_threads1"] = known_row
        row["known_policy95"] = impl.row_pass(known_row)
        row["stage"] = "known"
        if not row["known_policy95"]:
            row["reject"] = "known_below_policy95_or_runtime_hazard"
            screened.append(row)
            continue
        row["reject"] = None
        screened.append(row)
        row["data"] = data
        eligible.append(row)
        print(json.dumps({"i": index, "n": len(records), "task": row["task"],
                          "cost": profile["cost"], "authority": row["authority_cost"],
                          "known": known_row.get("accuracy"), "eligible": len(eligible)}),
              flush=True)
        if index % 50 == 0:
            print(json.dumps({"progress": index, "n": len(records),
                              "eligible": len(eligible)}), flush=True)

    grouped = defaultdict(list)
    for row in eligible:
        grouped[int(row["task"])].append(row)
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    fresh_results = []
    finalists = []
    candidate_dir = HERE / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for task, task_rows in sorted(grouped.items()):
        task_rows.sort(key=lambda item: (int(item["profile"]["cost"]), item["sha256"]))
        for rank, item in enumerate(task_rows[:5], 1):
            data = item["data"]
            known_cases, _ = known_cache[task]
            known_four = support.evaluate_four(data, known_cases)
            fresh_runs = []
            for seed in (402_200_000 + task, 402_300_000 + task):
                cases, generation = support.fresh_cases(task, seed, task_map)
                runtime_rows = support.evaluate_four(data, cases)
                fresh_runs.append({
                    "seed": seed,
                    "generation": generation,
                    "runtime": {name: impl.compact(value)
                                for name, value in runtime_rows.items()},
                    "pass": all(impl.row_pass(value) for value in runtime_rows.values()),
                })
            result = {key: value for key, value in item.items() if key != "data"}
            result["known_four"] = {name: impl.compact(value)
                                    for name, value in known_four.items()}
            result["known_four_pass"] = all(
                impl.row_pass(value) for value in known_four.values()
            )
            result["fresh"] = fresh_runs
            result["policy95_pass"] = bool(
                result["known_four_pass"] and all(run["pass"] for run in fresh_runs)
            )
            result["meets_half"] = bool(
                int(result["profile"]["cost"]) * 2 <= int(result["authority_cost"])
            )
            result["known_private_zero_lineage"] = task in PRIVATE_ZERO
            result["classification"] = (
                "POLICY95_PRIVATE_ZERO_RISK" if result["policy95_pass"] and task in PRIVATE_ZERO
                else "POLICY95_EMPIRICAL_NOT_GENERATOR_PROVEN" if result["policy95_pass"]
                else "REJECT"
            )
            fresh_results.append(result)
            print(json.dumps({"task": task, "rank": rank,
                              "cost": result["profile"]["cost"],
                              "fresh": [run["runtime"]["disable_threads1"]["accuracy"]
                                        for run in fresh_runs],
                              "pass": result["policy95_pass"]}), flush=True)
            if result["policy95_pass"]:
                target = candidate_dir / (
                    f"task{task:03d}_cost{result['profile']['cost']}_"
                    f"{result['sha256'][:12]}_POLICY95.onnx"
                )
                target.write_bytes(data)
                result["saved_path"] = str(target.relative_to(ROOT))
                finalists.append(result)
                break

    for row in eligible:
        row.pop("data", None)
    payload = {
        "authority": "submission_base_8011.05.zip",
        "authority_sha256": inventory["authority_sha256"],
        "threshold": 0.95,
        "source_inventory": str(INVENTORY.relative_to(ROOT)),
        "inventory_candidate_count": len(rows),
        "structure_prepass_count": len(records),
        "actual_strict_lower_structure_known95_count": len(eligible),
        "fresh_per_seed": 2000,
        "fresh_seeds_per_candidate": 2,
        "ort_configs": ["disable_threads1", "disable_threads4",
                        "default_threads1", "default_threads4"],
        "finalist_count": len(finalists),
        "guaranteed_safe_finalists": [],
        "policy95_finalists": finalists,
        "half_finalists": [row for row in finalists if row["meets_half"]],
        "strict_nonhalf_finalists": [row for row in finalists if not row["meets_half"]],
        "fresh_results": fresh_results,
        "screened": screened,
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "new evidence lane only; root and others untouched",
    }
    (HERE / "policy95_history_evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(records), "eligible": len(eligible),
                      "finalists": [{"task": row["task"],
                                      "cost": row["profile"]["cost"],
                                      "authority_cost": row["authority_cost"],
                                      "classification": row["classification"]}
                                     for row in finalists],
                      "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
