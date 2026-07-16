#!/usr/bin/env python3
"""Final fail-closed audit for B24 task256/task257."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE_ZIP = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave16_candidate_meta.zip"
EXPECTED_ZIP_SHA256 = "4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a"
EXPECTED_BASELINES = {
    256: ("cf7b3dbe9044cd993e4a5cbb2f3d7c85ae10d094f94a8e940809abdd223aa1f2", 119, 63),
    257: ("32e5452b9089ab217e13dc4aac064b7807f9397603a66e5a5a945ffa3b0f5ef6", 114, 27),
}
TASK_HASH = {256: "a65b410d", 257: "a68b268e"}
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))


def load_shared_audit():
    path = ROOT / "scripts/golf/loop_7999_13/lane_b15/audit_candidates.py"
    spec = importlib.util.spec_from_file_location("b24_shared_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def solve256(grid: list[list[int]]) -> list[list[int]]:
    """Readable literal expansion of raw/task256.py."""
    result = [row[:] for row in grid]
    red_row = max(result)
    length = sum(red_row) // 2
    triangle = row = result.index(red_row) + length
    for _ in range(row):
        target_row = row - triangle
        color = 2 + (triangle > length) - (triangle < length)
        result[target_row][:triangle] = [color] * triangle
        triangle -= 1
    return result


def solve257(grid: list[list[int]]) -> list[list[int]]:
    """Four fixed quadrants with generator overwrite priority 7>4>8>6."""
    return [
        [
            grid[row][col]
            or grid[row][col + 5]
            or grid[row + 5][col]
            or grid[row + 5][col + 5]
            for col in range(4)
        ]
        for row in range(4)
    ]


def rule_audit(task: int) -> dict[str, Any]:
    solver = solve256 if task == 256 else solve257
    examples = json.loads(
        (ROOT / "inputs/neurogolf-2026" / f"task{task:03d}.json").read_text()
    )
    subsets: dict[str, dict[str, int]] = {}
    right = wrong = 0
    for subset in ("train", "test", "arc-gen"):
        subset_right = subset_wrong = 0
        for pair in examples[subset]:
            if solver(pair["input"]) == pair["output"]:
                right += 1
                subset_right += 1
            else:
                wrong += 1
                subset_wrong += 1
        subsets[subset] = {"right": subset_right, "wrong": subset_wrong}

    module = importlib.import_module(f"task_{TASK_HASH[task]}")
    random.seed(24_000_000 + task)
    fresh_right = fresh_wrong = generation_errors = 0
    for _ in range(5000):
        try:
            pair = module.generate()
            if solver(pair["input"]) == pair["output"]:
                fresh_right += 1
            else:
                fresh_wrong += 1
        except Exception:
            generation_errors += 1
    return {
        "task": task,
        "generator_hash": TASK_HASH[task],
        "known_right": right,
        "known_wrong": wrong,
        "known_subsets": subsets,
        "fresh_right": fresh_right,
        "fresh_wrong": fresh_wrong,
        "fresh_generation_errors": generation_errors,
        "classification": "D_global_geometry" if task == 256 else "A_fixed_four_source_priority",
        "rule": (
            "find the red prefix of length L at row R; let T=R+L; draw green prefixes "
            "of lengths T..L+1 above, keep the red length-L row, then blue prefixes "
            "L-1..1 below"
            if task == 256
            else "for each output cell choose the first nonzero among TL, TR, BL, BR at offsets 0/5"
        ),
    }


def baseline_audit(shared: Any, task: int, payload: bytes) -> dict[str, Any]:
    expected_sha, expected_cost, expected_max = EXPECTED_BASELINES[task]
    digest = sha256_bytes(payload)
    if digest != expected_sha:
        raise RuntimeError(f"task{task} baseline SHA mismatch: {digest}")
    path = HERE / f"baseline_task{task:03d}.onnx"
    path.write_bytes(payload)
    model = onnx.load_model_from_string(payload)
    structure = shared.structural(copy.deepcopy(model))
    runtime_shape = shared.trace_runtime_shapes(copy.deepcopy(model), task)
    known = shared.known_dual(copy.deepcopy(model), task)
    actual = shared.actual_score(copy.deepcopy(model), task, f"baseline_{task}")
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    if actual is None or actual["cost"] != expected_cost or max_einsum != expected_max:
        raise RuntimeError(f"task{task} baseline cost/structure mismatch")
    non_giant_checks = {
        key: value for key, value in structure["checks"].items() if key != "no_giant_einsum"
    }
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "actual_score": actual,
        "max_einsum_inputs": max_einsum,
        "existing_giant_einsum": not structure["checks"]["no_giant_einsum"],
        "all_non_giant_structural_checks": all(non_giant_checks.values()),
        "structure": structure,
        "runtime_shape": runtime_shape,
        "known_dual": known,
        "known_perfect": all(
            row["wrong"] == 0 and row["errors"] == 0 for row in known
        ),
    }


def history_summary() -> dict[str, Any]:
    evidence = json.loads((HERE / "history_inventory.json").read_text())
    rows = evidence["loose_rows"]
    summary: dict[str, Any] = {}
    for task in (256, 257):
        _, baseline_cost, baseline_max = EXPECTED_BASELINES[task]
        task_rows = [row for row in rows if row["task"] == task]
        below = [
            row
            for row in task_rows
            if isinstance(row.get("cost"), int) and 0 <= row["cost"] < baseline_cost
        ]
        invalid_negative = [
            row
            for row in task_rows
            if isinstance(row.get("cost"), int) and row["cost"] < 0
        ]
        summary[str(task)] = {
            "loose_files_seen": evidence["loose_files_seen"][str(task)],
            "distinct_hashes": evidence["loose_distinct_hashes"][str(task)],
            "valid_static_below_baseline": [
                {
                    "sha256": row["sha256"],
                    "cost": row["cost"],
                    "max_einsum_inputs": row["max_einsum_inputs"],
                    "enlarges_giant": row["max_einsum_inputs"] > baseline_max,
                    "source": row["loose_sources"][0],
                }
                for row in below
            ],
            "invalid_negative_cost_rows": [
                {"sha256": row["sha256"], "source": row["loose_sources"][0]}
                for row in invalid_negative
            ],
        }
    return summary


def main() -> int:
    archive_digest = hashlib.sha256(BASELINE_ZIP.read_bytes()).hexdigest()
    if archive_digest != EXPECTED_ZIP_SHA256:
        raise RuntimeError(f"Wave16 SHA mismatch: {archive_digest}")
    shared = load_shared_audit()
    rules = [rule_audit(task) for task in (256, 257)]
    if any(row["known_wrong"] or row["fresh_wrong"] or row["fresh_generation_errors"] for row in rules):
        raise RuntimeError("readable rule audit failed")
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        baselines = [
            baseline_audit(shared, task, archive.read(f"task{task:03d}.onnx"))
            for task in (256, 257)
        ]
    candidate_screen = json.loads((HERE / "candidate_screen.json").read_text())
    if candidate_screen["known_survivors"] != 0:
        raise RuntimeError("candidate survived fail-fast known screen; full audit required")
    history = history_summary()

    payload = {
        "baseline_zip": str(BASELINE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": archive_digest,
        "tasks": [256, 257],
        "rules": rules,
        "baselines": baselines,
        "history": history,
        "candidate_count": candidate_screen["candidate_count"],
        "candidate_known_survivors": 0,
        "candidate_screen": str((HERE / "candidate_screen.json").relative_to(ROOT)),
        "winners": [],
        "verified_gain": 0.0,
        "fresh5000_candidate_run": False,
        "fresh5000_candidate_skip_reason": (
            "All 31 strictly cheaper probes fail the first known example in both ORT modes "
            "or raise a runtime diagonal-shape error; no candidate is eligible for fresh validation."
        ),
    }
    (HERE / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    winner_manifest = {
        "baseline_zip_sha256": archive_digest,
        "winners": [],
        "winner_count": 0,
        "verified_gain": 0.0,
        "reason": "No below-incumbent task256/task257 model passes the strict known gate.",
        "audit": str((HERE / "audit.json").relative_to(ROOT)),
    }
    (HERE / "winner_manifest.json").write_text(
        json.dumps(winner_manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "rules": [(row["task"], row["known_right"], row["fresh_right"]) for row in rules],
                "baselines": [
                    (
                        row["task"],
                        row["actual_score"]["cost"],
                        row["known_perfect"],
                        row["runtime_shape"]["shape_cloak"],
                        row["max_einsum_inputs"],
                    )
                    for row in baselines
                ],
                "candidate_count": candidate_screen["candidate_count"],
                "candidate_known_survivors": 0,
                "winners": [],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
