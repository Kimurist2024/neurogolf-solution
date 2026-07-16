#!/usr/bin/env python3
"""Direct generator-rule evidence for task023 non-injectivity and task036 crop."""

from __future__ import annotations

import importlib
import json
import random
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/scratch_codex/task036"))

from reference_solver import solve as solve036  # noqa: E402


def freeze(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


def task023_conflict() -> dict[str, object]:
    generator = importlib.import_module("task_150deff5")
    cases = []
    for seed in (29685, 120072):
        random.seed(seed)
        cases.append(generator.generate())
    same_input = freeze(cases[0]["input"]) == freeze(cases[1]["input"])
    different_output = freeze(cases[0]["output"]) != freeze(cases[1]["output"])
    return {
        "seeds": [29685, 120072],
        "same_input": same_input,
        "different_output": different_output,
        "proof": same_input and different_output,
        "shape": [len(cases[0]["input"]), len(cases[0]["input"][0])],
        "input": cases[0]["input"],
        "output_seed_29685": cases[0]["output"],
        "output_seed_120072": cases[1]["output"],
        "consequence": "No deterministic input-only ONNX can implement both legal generator outputs.",
    }


def task036_reference() -> dict[str, object]:
    examples = json.loads((ROOT / "inputs/neurogolf-2026/task036.json").read_text())
    fixture_right = fixture_wrong = 0
    first_failure = None
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[subset]):
            got = solve036(example["input"])
            if got == example["output"]:
                fixture_right += 1
            else:
                fixture_wrong += 1
                first_failure = first_failure or {"subset": subset, "index": index}

    generator = importlib.import_module("task_1f85a75f")
    seed = 150360036
    random.seed(seed)
    requested = 5000
    fresh_right = fresh_wrong = errors = 0
    started = time.monotonic()
    for index in range(requested):
        try:
            example = generator.generate()
            got = solve036(example["input"])
            if got == example["output"]:
                fresh_right += 1
            else:
                fresh_wrong += 1
                first_failure = first_failure or {
                    "fresh_index": index,
                    "input": example["input"],
                    "expected": example["output"],
                    "got": got,
                }
        except Exception as exc:  # noqa: BLE001 - direct rule errors are evidence
            errors += 1
            first_failure = first_failure or {"fresh_index": index, "error": repr(exc)}
    return {
        "rule": (
            "Select the compact connected nonzero color whose full support lies in a <=5x5 bbox "
            "with a clear moat, then return the complete input crop at that tight bbox."
        ),
        "fixture_right": fixture_right,
        "fixture_wrong": fixture_wrong,
        "fresh_seed": seed,
        "fresh_requested": requested,
        "fresh_right": fresh_right,
        "fresh_wrong": fresh_wrong,
        "errors": errors,
        "first_failure": first_failure,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    result = {
        "task023": task023_conflict(),
        "task036": task036_reference(),
    }
    (HERE / "generator_rule_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task023_conflict": result["task023"]["proof"],
                "task036": {
                    key: result["task036"][key]
                    for key in ("fixture_right", "fixture_wrong", "fresh_right", "fresh_wrong", "errors")
                },
            },
            indent=2,
        )
    )
    return 0 if result["task023"]["proof"] and not result["task036"]["fixture_wrong"] and not result["task036"]["fresh_wrong"] and not result["task036"]["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
