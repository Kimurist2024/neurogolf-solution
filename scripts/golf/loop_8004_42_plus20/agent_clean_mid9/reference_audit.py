#!/usr/bin/env python3
"""Independent generator-rule reference audit for the four mid9 tasks."""

from __future__ import annotations

import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(TASK_DIR))

SEEDS = (260714009, 910714009)
COUNT = 50
HASHES = {88: "3de23699", 89: "3e980e27", 191: "7df24a62"}
SOLVERS = {
    88: ROOT / "scripts/golf/scratch/task088/ref_verify.py",
    89: ROOT / "scripts/golf/scratch_claude/task089/solver.py",
    191: ROOT / "scripts/golf/scratch_claude/task191/solver.py",
}


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task002_counterexample() -> dict:
    module = load_path(
        "task002_spec_check_mid9",
        ROOT / "scripts/golf/scratch_codex/task002/spec_check.py",
    )
    a_params = dict(
        size=8,
        rows=[7],
        cols=[0],
        brows=[0, 3],
        bcols=[0, 3],
        wides=[4, 3],
        talls=[3, 3],
    )
    b_params = dict(
        size=8,
        rows=[1, 1, 0, 0, 2, 2, 7],
        cols=[0, 3, 1, 2, 1, 2, 0],
        brows=[3],
        bcols=[3],
        wides=[3],
        talls=[3],
    )
    a = module.literal_from_params(**a_params)
    b = module.literal_from_params(**b_params)
    diffs = [
        [r, c, a["output"][r][c], b["output"][r][c]]
        for r in range(8)
        for c in range(8)
        if a["output"][r][c] != b["output"][r][c]
    ]
    return {
        "same_input": a["input"] == b["input"],
        "different_output_cells": diffs,
        "conclusion": "no deterministic input-only exact function exists",
    }


def audit_task(task: int) -> dict:
    generator = importlib.import_module(f"task_{HASHES[task]}")
    solver = load_path(f"solver_{task}_mid9", SOLVERS[task]).solve
    rows = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        right = 0
        first_failure = None
        for index in range(COUNT):
            case = generator.generate()
            got = np.asarray(solver(case["input"]), dtype=np.int64)
            want = np.asarray(case["output"], dtype=np.int64)
            if np.array_equal(got, want):
                right += 1
            elif first_failure is None:
                first_failure = {
                    "index": index,
                    "different_cells": int(np.count_nonzero(got != want))
                    if got.shape == want.shape
                    else None,
                }
        rows.append(
            {
                "seed": seed,
                "right": right,
                "total": COUNT,
                "accuracy": right / COUNT,
                "first_failure": first_failure,
            }
        )
    return {"task": task, "hash": HASHES[task], "seeds": rows}


def main() -> None:
    result = {
        "task002_input_non_determinism": task002_counterexample(),
        "reference_rules": [audit_task(task) for task in (88, 89, 191)],
    }
    (HERE / "reference_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
