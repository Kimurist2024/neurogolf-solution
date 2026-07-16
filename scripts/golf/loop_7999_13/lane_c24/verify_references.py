#!/usr/bin/env python3
"""Verify readable generator-derived references for C24 tasks."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(TASK_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPEC363 = load_module(
    "task363_spec_reference",
    ROOT / "scripts" / "golf" / "scratch_codex" / "task363" / "spec_reference.py",
)
GEN363 = SPEC363.load_task()
GEN388 = load_module("task_f5b8619d_c24", TASK_DIR / "task_f5b8619d.py")


def solve388(grid: list[list[int]]) -> list[list[int]]:
    array = np.asarray(grid, dtype=np.uint8)
    nonzero = array != 0
    columns = nonzero.any(axis=0)
    color = int(array[nonzero][0])
    tile = np.zeros_like(array)
    tile[:, columns] = 8
    tile[nonzero] = color
    return np.tile(tile, (2, 2)).tolist()


def visible363() -> tuple[int, list[dict[str, object]]]:
    examples = json.loads(
        (ROOT / "inputs" / "neurogolf-2026" / "task363.json").read_text()
    )
    total = 0
    mismatches: list[dict[str, object]] = []
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[subset]):
            actual = SPEC363.solve(example["input"])
            expected = np.asarray(example["output"], dtype=np.uint8)
            if not np.array_equal(actual, expected):
                mismatches.append({"subset": subset, "index": index})
            total += 1
    return total, mismatches


def visible388() -> int:
    examples = json.loads(
        (ROOT / "inputs" / "neurogolf-2026" / "task388.json").read_text()
    )
    total = 0
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[subset]):
            if solve388(example["input"]) != example["output"]:
                raise AssertionError(f"task388 visible mismatch {subset}[{index}]")
            total += 1
    return total


def fresh363(count: int) -> None:
    for seed in range(count):
        random.seed(seed)
        example = GEN363.generate()
        if not np.array_equal(
            SPEC363.solve(example["input"]),
            np.asarray(example["output"], dtype=np.uint8),
        ):
            raise AssertionError(f"task363 fresh mismatch seed={seed}")


def fresh388(count: int) -> None:
    for seed in range(count):
        random.seed(seed)
        example = GEN388.generate()
        if solve388(example["input"]) != example["output"]:
            raise AssertionError(f"task388 fresh mismatch seed={seed}")


def main() -> None:
    count363, mismatches363 = visible363()
    assert mismatches363 == [
        {"subset": "train", "index": 0},
        {"subset": "train", "index": 1},
    ]
    fresh363(5000)
    SPEC363.prove_non_identifiability(GEN363)

    count388 = visible388()
    fresh388(5000)

    evidence = {
        "task363": {
            "generator": "inputs/arc-gen-repo/tasks/task_e5062a87.py",
            "reference": "scripts/golf/scratch_codex/task363/spec_reference.py",
            "visible": {
                "checked": count363,
                "right": count363 - len(mismatches363),
                "wrong": len(mismatches363),
                "errors": 0,
                "mismatches": mismatches363,
            },
            "fresh": {"right": 5000, "wrong": 0, "errors": 0},
            "non_identifiable": True,
            "non_identifiability_reason": (
                "The same validate train[1] input is produced by a legal complete "
                "three-location parameterization with a different output."
            ),
        },
        "task388": {
            "generator": "inputs/arc-gen-repo/tasks/task_f5b8619d.py",
            "reference": "scripts/golf/loop_7999_13/lane_c24/verify_references.py::solve388",
            "visible": {"right": count388, "wrong": 0, "errors": 0},
            "fresh": {"right": 5000, "wrong": 0, "errors": 0},
        },
        "fresh_seed_policy": "Reset Python's random module to each integer 0..4999.",
    }
    (HERE / "reference_audit.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"task363 visible={count363 - len(mismatches363)}/{count363} "
        "fresh=5000/5000 non_identifiable=true"
    )
    print(f"task388 visible={count388}/{count388} fresh=5000/5000")


if __name__ == "__main__":
    main()
