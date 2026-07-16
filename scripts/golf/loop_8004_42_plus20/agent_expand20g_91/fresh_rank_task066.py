#!/usr/bin/env python3
"""Independent local fresh classification for the four task066 probes.

Fresh evidence is only a local rejection/ranking signal.  It never upgrades a
candidate to fixed or LB-white.  Exact SHA matches to prior LB-black payloads
remain black regardless of the local result.
"""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COUNT_PER_SEED = 500
SEEDS = (91_066_091, 91_066_092)
FRESH_THRESHOLD = 0.90

# These are byte-exact payload matches, not task-level exclusions.  The cited
# histories all reverted the exact payload after LB diagnosis.
KNOWN_LB_BLACK = {
    "d909159d16436ceea64bfe4a97ebd27a058032cda478e9359af12ef8992f0470": {
        "lineage": "others/2/1200/task066_cost368_improved.onnx",
        "evidence": (
            "Claude transcript 5e3139b8... lines 11210/11221/11249/11357: "
            "1200 A2 deficit identifies task066@cost368; black-set accounting "
            "closes and submission_1200final reverts it"
        ),
    },
    "65a1b5888e491ac26b4c3cdb07436295939835e3a79f9b71b7eb83dccd01152a": {
        "lineage": "others/2/1102/task066_cost582_improved.onnx",
        "evidence": (
            "memory/current-best.md: others/1102 task066 is single-task-probe "
            "BLACK; source/archive SHA is byte-exact"
        ),
    },
    "349ea2636f3398c808277adb110b607e1f1fc42b2257ec75018eb162f680d512": {
        "lineage": "others/2/1101/task066_super_improved.onnx",
        "evidence": (
            "memory/current-best.md: others/1101 task066 is single-task-probe "
            "BLACK; source/archive SHA is byte-exact"
        ),
    },
}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from scripts.lib import scoring  # noqa: E402


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize rejected")
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def make_cases(module: object, seed: int) -> tuple[list[dict[str, np.ndarray]], int, int]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    cases: list[dict[str, np.ndarray]] = []
    attempts = generation_errors = 0
    while len(cases) < COUNT_PER_SEED and attempts < COUNT_PER_SEED * 20:
        attempts += 1
        try:
            raw = module.generate()
            converted = scoring.convert_to_numpy(raw)
            if converted is not None:
                cases.append(converted)
        except Exception:  # noqa: BLE001
            generation_errors += 1
    if len(cases) != COUNT_PER_SEED:
        raise RuntimeError(
            f"seed {seed}: generated only {len(cases)}/{COUNT_PER_SEED} valid cases"
        )
    return cases, attempts, generation_errors


def evaluate(
    runtime: ort.InferenceSession, cases: list[dict[str, np.ndarray]]
) -> dict[str, int | float | None]:
    right = wrong = errors = 0
    first_failure = None
    for index, case in enumerate(cases, start=1):
        try:
            output = runtime.run(
                [runtime.get_outputs()[0].name],
                {runtime.get_inputs()[0].name: case["input"]},
            )[0]
            correct = np.array_equal(output > 0.0, case["output"].astype(bool))
            right += int(correct)
            wrong += int(not correct)
            if not correct and first_failure is None:
                first_failure = index
        except Exception:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = index
    return {
        "right": right,
        "wrong": wrong,
        "runtime_errors": errors,
        "accuracy": right / len(cases),
        "first_failure": first_failure,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    module = importlib.import_module(f"task_{task_map['066']}")
    case_sets = []
    for seed in SEEDS:
        cases, attempts, generation_errors = make_cases(module, seed)
        case_sets.append(
            {
                "seed": seed,
                "cases": cases,
                "valid": len(cases),
                "attempts": attempts,
                "generation_errors": generation_errors,
            }
        )

    probe_path = HERE / "probe_manifest.json"
    original_manifest = json.loads(probe_path.read_text())
    original_probes = original_manifest["candidates"]
    rows = []
    for probe in original_probes:
        if int(probe["task"]) != 66:
            continue
        model = onnx.load(ROOT / probe["path"])
        runtimes = {
            "disable_all": session(model, True),
            "default": session(model, False),
        }
        seed_rows = []
        accuracies = []
        for case_set in case_sets:
            modes = {}
            for label, runtime in runtimes.items():
                mode_result = evaluate(runtime, case_set["cases"])
                modes[label] = mode_result
                accuracies.append(float(mode_result["accuracy"]))
            seed_rows.append(
                {
                    "seed": case_set["seed"],
                    "valid": case_set["valid"],
                    "attempts": case_set["attempts"],
                    "generation_errors": case_set["generation_errors"],
                    "modes": modes,
                }
            )
        minimum_seed_accuracy = min(accuracies)
        sha256 = probe["sha256"]
        if sha256 in KNOWN_LB_BLACK:
            decision = "KNOWN_LB_BLACK"
        elif minimum_seed_accuracy < FRESH_THRESHOLD:
            decision = "REJECT_LOCAL_FALSE_ACCEPT"
        else:
            decision = "LB_PROBE_REQUIRED"
        row = {
            "task": 66,
            "path": probe["path"],
            "sha256": sha256,
            "projected_gain": probe["projected_gain"],
            "seeds": seed_rows,
            "minimum_seed_accuracy": minimum_seed_accuracy,
            "known_lb_black_exact_sha": KNOWN_LB_BLACK.get(sha256),
            "decision": decision,
            "policy": (
                "fresh is local rejection/ranking only; exact prior LB-black SHA "
                "overrides local accuracy; no local result creates fixed/LB-white status"
            ),
        }
        rows.append(row)
        print(
            sha256[:12],
            f"min_accuracy={minimum_seed_accuracy:.4f}",
            decision,
            flush=True,
        )

    rows.sort(key=lambda row: (-row["minimum_seed_accuracy"], -row["projected_gain"]))
    evidence = {
        "task": 66,
        "seeds": list(SEEDS),
        "count_per_seed": COUNT_PER_SEED,
        "fresh_threshold": FRESH_THRESHOLD,
        "known_lb_black_exact_sha": KNOWN_LB_BLACK,
        "task_history_risk": (
            "task066 has repeated LB-black outcomes across distinct nets; this is a "
            "risk note, not a task-level automatic exclusion"
        ),
        "policy": "local fresh is rejection/ranking information only",
        "rows": rows,
    }
    output = HERE / "evidence/task066_fresh_2seed_500.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    by_sha = {row["sha256"]: row for row in rows}

    result_path = HERE / "result.json"
    result = json.loads(result_path.read_text())
    for row in result["rows"]:
        fresh = by_sha.get(row["sha256"])
        if fresh is not None:
            row["local_fresh_classification"] = fresh
            row["decision"] = fresh["decision"]
    result["lb_probe_required"] = [
        row for row in result["rows"] if row["decision"] == "LB_PROBE_REQUIRED"
    ]
    result["known_lb_black"] = [
        row for row in result["rows"] if row["decision"] == "KNOWN_LB_BLACK"
    ]
    result["local_false_accept"] = [
        row
        for row in result["rows"]
        if row["decision"] == "REJECT_LOCAL_FALSE_ACCEPT"
    ]
    result["decision_counts"] = dict(Counter(row["decision"] for row in result["rows"]))
    result["local_fresh_classification_evidence"] = str(output.relative_to(ROOT))
    result["task066_exact_sha_catalog"] = KNOWN_LB_BLACK
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    candidates = []
    for probe in original_probes:
        fresh = by_sha.get(probe["sha256"])
        if fresh is not None and fresh["decision"] == "LB_PROBE_REQUIRED":
            updated = dict(probe)
            updated["local_fresh_classification"] = fresh
            candidates.append(updated)
    manifest = {
        "authority": original_manifest["authority"],
        "authority_sha256": original_manifest["authority_sha256"],
        "status": "LB_PROBE_REQUIRED" if candidates else "NO_NEW_LB_PROBES",
        "candidates": candidates,
        "known_lb_black": [row for row in rows if row["decision"] == "KNOWN_LB_BLACK"],
        "local_false_accept": [
            row for row in rows if row["decision"] == "REJECT_LOCAL_FALSE_ACCEPT"
        ],
        "local_fresh_policy": "ranking/rejection only; never fixed adoption evidence",
        "merge_performed": False,
    }
    probe_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
