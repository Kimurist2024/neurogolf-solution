#!/usr/bin/env python3
"""Independently prove Sakana true-rule functions on known and fresh cases."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import multiprocessing as mp
import random
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (125, 145, 187, 192, 196, 204, 208, 340, 344)
HASH_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())


def normalize(value):
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return int(value) if isinstance(value, bool) else value


def audit_one(job: tuple[int, int, int]) -> dict[str, object]:
    task, fresh_count, seed = job
    sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
    sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
    rule = runpy.run_path(str(ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"))["p"]
    generator = importlib.import_module(f"task_{HASH_MAP[f'{task:03d}']}")
    known = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    known_examples = known["train"] + known["test"] + known["arc-gen"]

    def check(example: dict[str, object]) -> bool:
        actual = normalize(rule(copy.deepcopy(example["input"])))
        return actual == normalize(example["output"])

    known_right = sum(check(example) for example in known_examples)
    random.seed(seed + task)
    fresh_right = generation_errors = rule_errors = 0
    first_wrong = None
    generated = 0
    while generated < fresh_count:
        try:
            example = generator.generate()
        except Exception as exc:  # noqa: BLE001
            generation_errors += 1
            if generation_errors > fresh_count:
                raise RuntimeError("generator repeatedly failed") from exc
            continue
        generated += 1
        try:
            correct = check(example)
        except Exception as exc:  # noqa: BLE001
            rule_errors += 1
            if first_wrong is None:
                first_wrong = {"index": generated - 1, "error": repr(exc)}
            continue
        fresh_right += int(correct)
        if not correct and first_wrong is None:
            first_wrong = {
                "index": generated - 1,
                "input_shape": [len(example["input"]), len(example["input"][0])],
            }
    return {
        "task": task,
        "known_right": known_right,
        "known_total": len(known_examples),
        "known_perfect": known_right == len(known_examples),
        "fresh_right": fresh_right,
        "fresh_total": generated,
        "fresh_accuracy": fresh_right / generated,
        "generation_errors": generation_errors,
        "rule_errors": rule_errors,
        "first_wrong": first_wrong,
        "seed": seed + task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=88_000_000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = HERE / "true_rule_audit.json"
    rows: list[dict[str, object]] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(
            audit_one, [(task, args.fresh, args.seed) for task in TASKS]
        ):
            rows.append(row)
            rows.sort(key=lambda item: int(item["task"]))
            output.write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
