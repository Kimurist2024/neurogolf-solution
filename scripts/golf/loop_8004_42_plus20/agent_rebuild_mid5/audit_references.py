#!/usr/bin/env python3
"""Audit independent Sakana true-rule references against generator truth."""

from __future__ import annotations

import copy
import importlib
import json
import random
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
HASHES = {23: "150deff5", 187: "7b6016b9", 209: "8a004b2b", 367: "e73095fd"}


def normalized(value):
    if isinstance(value, (list, tuple, bytes, bytearray)):
        return [normalized(item) for item in value]
    return int(value) if isinstance(value, bool) else value


def main() -> None:
    sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
    common = importlib.import_module("common")
    report = {}
    for task, hash_value in HASHES.items():
        rule = runpy.run_path(
            str(ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py")
        )["p"]
        generator = importlib.import_module(f"task_{hash_value}")
        data = json.loads(
            (ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text()
        )
        known = [
            item
            for split in ("train", "test", "arc-gen")
            for item in data.get(split, [])
        ]

        def correct(example) -> bool:
            actual = normalized(rule(copy.deepcopy(example["input"])))
            return actual == normalized(example["output"])

        known_right = sum(correct(item) for item in known)
        streams = []
        for base_seed in (930_000_000, 940_000_000):
            right = 0
            first_failure = None
            for index in range(2000):
                seed = base_seed + task * 10_000 + index
                random.seed(seed)
                common.random.seed(seed)
                example = generator.generate()
                ok = correct(example)
                right += int(ok)
                if not ok and first_failure is None:
                    first_failure = {"index": index, "seed": seed}
            streams.append(
                {
                    "base_seed": base_seed,
                    "right": right,
                    "total": 2000,
                    "accuracy": right / 2000,
                    "first_failure": first_failure,
                }
            )
        report[str(task)] = {
            "known_right": known_right,
            "known_total": len(known),
            "streams": streams,
        }
        print(task, report[str(task)], flush=True)
    (HERE / "reference_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
