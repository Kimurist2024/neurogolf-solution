#!/usr/bin/env python3
"""Fail-closed B25 audit for task232 and task369."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import random
import sys
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE_ZIP = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave16_candidate_meta.zip"
EXPECTED_ZIP_SHA256 = "4014cbafea4862f67ebf5ff24be13149b45b333c95bfa680be7216f001a6bb3a"
EXPECTED_BASELINES = {
    232: ("675073205692bd769bb45a2b4e3f8608ea68af838257974b2bf5f65c0fa552ad", 116, 11),
    369: ("1630f7d64c2d7b99e107830d7639ff0b99b0eba0f1f33b09d0d411c7918e832e", 130, 0),
}
TASK_HASH = {232: "97999447", 369: "e8593010"}
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))


def shared_audit():
    path = ROOT / "scripts/golf/loop_7999_13/lane_b15/audit_candidates.py"
    spec = importlib.util.spec_from_file_location("b25_shared_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def solve232(grid: list[list[int]]) -> list[list[int]]:
    out = [row[:] for row in grid]
    for row_index, row in enumerate(grid):
        colored = [(col, value) for col, value in enumerate(row) if value != 0]
        if not colored:
            continue
        if len(colored) != 1:
            raise ValueError("generator invariant violated: multiple source cells in one row")
        start, color = colored[0]
        for col in range(start, len(row)):
            out[row_index][col] = color if (col - start) % 2 == 0 else 5
    return out


def solve369(grid: list[list[int]]) -> list[list[int]]:
    height, width = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    pending = {(row, col) for row in range(height) for col in range(width) if grid[row][col] == 0}
    while pending:
        start = pending.pop()
        component = [start]
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + dr, col + dc)
                if neighbor in pending:
                    pending.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        if not 1 <= len(component) <= 3:
            raise ValueError(f"generator invariant violated: component size {len(component)}")
        color = 4 - len(component)
        for row, col in component:
            out[row][col] = color
    return out


def rule_audit(task: int) -> dict[str, Any]:
    solver = solve232 if task == 232 else solve369
    examples = json.loads((ROOT / "inputs/neurogolf-2026" / f"task{task:03d}.json").read_text())
    subsets: dict[str, dict[str, int]] = {}
    known_right = known_wrong = 0
    for subset in ("train", "test", "arc-gen"):
        right = wrong = 0
        for pair in examples[subset]:
            if solver(pair["input"]) == pair["output"]:
                right += 1
                known_right += 1
            else:
                wrong += 1
                known_wrong += 1
        subsets[subset] = {"right": right, "wrong": wrong}

    module = importlib.import_module(f"task_{TASK_HASH[task]}")
    random.seed(25_000_000 + task)
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
        "known_right": known_right,
        "known_wrong": known_wrong,
        "known_subsets": subsets,
        "fresh_right": fresh_right,
        "fresh_wrong": fresh_wrong,
        "fresh_generation_errors": generation_errors,
        "classification": "A_row_local_finite_state" if task == 232 else "B_bounded_local_component",
        "rule": (
            "from the sole colored cell in each occupied row, alternate source color and gray 5 to the right"
            if task == 232
            else "recolor each orthogonally connected black component by size: 1->3, 2->2, 3->1"
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
    try:
        runtime = shared.trace_runtime_shapes(copy.deepcopy(model), task)
    except Exception as error:
        runtime = {"shape_cloak": None, "trace_error": f"{type(error).__name__}: {error}"}
    known = shared.known_dual(copy.deepcopy(model), task)
    score = shared.actual_score(copy.deepcopy(model), task, f"b25_baseline_{task}")
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    if score is None or score["cost"] != expected_cost or max_einsum != expected_max:
        raise RuntimeError(f"task{task} baseline cost/structure mismatch: {score}, max={max_einsum}")
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "actual_score": score,
        "max_einsum_inputs": max_einsum,
        "structure": structure,
        "runtime_shape": runtime,
        "known_dual": known,
        "known_perfect": all(row["wrong"] == 0 and row["errors"] == 0 for row in known),
        "disabled_known_perfect": known[0]["wrong"] == 0 and known[0]["errors"] == 0,
    }


def history_summary() -> dict[str, Any]:
    evidence = json.loads((HERE / "history_inventory.json").read_text())
    summary: dict[str, Any] = {}
    for task in (232, 369):
        base = EXPECTED_BASELINES[task][1]
        rows = [row for row in evidence["loose_rows"] if row["task"] == task]
        below = [row for row in rows if isinstance(row.get("cost"), int) and 0 <= row["cost"] < base]
        ties = [row for row in rows if row.get("cost") == base]
        summary[str(task)] = {
            "loose_files_seen": evidence["loose_files_seen"][str(task)],
            "distinct_hashes": evidence["loose_distinct_hashes"][str(task)],
            "valid_static_below_baseline": [
                {"sha256": row["sha256"], "cost": row["cost"], "source": row["loose_sources"][0]}
                for row in below
            ],
            "equal_cost_hashes": [
                {"sha256": row["sha256"], "cost": row["cost"], "source": row["loose_sources"][0]}
                for row in ties
            ],
        }
    return summary


def main() -> int:
    archive_digest = hashlib.sha256(BASELINE_ZIP.read_bytes()).hexdigest()
    if archive_digest != EXPECTED_ZIP_SHA256:
        raise RuntimeError(f"Wave16 SHA mismatch: {archive_digest}")
    rules = [rule_audit(task) for task in (232, 369)]
    if any(row["known_wrong"] or row["fresh_wrong"] or row["fresh_generation_errors"] for row in rules):
        raise RuntimeError("readable generator rule audit failed")
    shared = shared_audit()
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        baselines = [baseline_audit(shared, task, archive.read(f"task{task:03d}.onnx")) for task in (232, 369)]
    # The inherited task369 incumbent is itself invalid under default ORT
    # because of its shape-cloaked CenterCropPad declarations.  Preserve that
    # as rejection evidence; it does not relax the errors==0 requirement for
    # any new candidate.  Disabled ORT must still establish its historical
    # semantics, while task232 must pass both modes.
    if not baselines[0]["known_perfect"] or not baselines[1]["disabled_known_perfect"]:
        raise RuntimeError("baseline semantic reference audit failed")
    screen = json.loads((HERE / "candidate_screen.json").read_text())
    if screen["known_survivors"]:
        raise RuntimeError("a task232 probe survived and requires full candidate audit")
    history = history_summary()
    if any(history[str(task)]["valid_static_below_baseline"] for task in (232, 369)):
        raise RuntimeError("unresolved historical below-baseline candidate")

    trained = json.loads((HERE / "rank3_training.json").read_text())
    payload = {
        "baseline_zip": str(BASELINE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": archive_digest,
        "tasks": [232, 369],
        "rules": rules,
        "baselines": baselines,
        "history": history,
        "task232_probe_count": screen["candidate_count"],
        "task232_probe_known_survivors": screen["known_survivors"],
        "task232_rank3_training": {
            "exhaustive_atomic_rows": trained["exhaustive_atomic_rows"],
            "best_wrong_atomic_cells": trained["best"]["wrong"],
            "candidate": trained["candidate"],
            "decision": "reject_generator_exhaustive_atomic_gate",
        },
        "task369_decision": (
            "reject all incumbent/tie families as shape-cloaked (the Wave16 incumbent also has one default-ORT session error); "
            "complete loose/archive history has no candidate below 130, and a truthful local-rule graph necessarily exposes "
            "nontrivial 30x30 intermediates"
        ),
        "winners": [],
        "verified_gain": 0.0,
        "fresh5000_candidate_run": False,
        "fresh5000_candidate_skip_reason": (
            "No strictly cheaper candidate passed the complete known/generator-exhaustive gate; fresh5000 was run on both readable true rules."
        ),
    }
    (HERE / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    manifest = {
        "baseline_zip_sha256": archive_digest,
        "winners": [],
        "winner_count": 0,
        "verified_gain": 0.0,
        "reason": "No strictly cheaper truthful task232/task369 model passes the strict semantic and shape-safety gates.",
        "audit": str((HERE / "audit.json").relative_to(ROOT)),
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "rules": [(row["task"], row["known_right"], row["fresh_right"]) for row in rules],
        "baselines": [(row["task"], row["actual_score"]["cost"], row["known_perfect"], row["runtime_shape"].get("shape_cloak")) for row in baselines],
        "probe_count": screen["candidate_count"],
        "winners": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
