#!/usr/bin/env python3
"""Prove the decoded Sakana rules against stored and fresh generator outputs."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import runpy
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (46, 157, 161, 189, 384, 193, 195, 281)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def same(actual: object, expected: object) -> bool:
    return np.array_equal(np.asarray(actual), np.asarray(expected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=int, default=2000)
    parser.add_argument("--tasks", nargs="*", type=int, default=list(TASKS))
    args = parser.parse_args()
    mapping = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    audit_path = HERE / "true_rule_audit.json"
    if audit_path.exists():
        output = json.loads(audit_path.read_text())
    else:
        output = {"fresh_per_seed": args.fresh, "tasks": {}}
    output["fresh_per_seed"] = args.fresh
    for task in args.tasks:
        rule = runpy.run_path(str(ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"))["p"]
        generator = importlib.import_module(f"task_{mapping[f'{task:03d}']}")
        known = scoring.load_examples(task)
        known_stats = {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
        for split in ("train", "test", "arc-gen"):
            for index, example in enumerate(known[split]):
                try:
                    actual = rule(copy.deepcopy(example["input"]))
                    if same(actual, example["output"]):
                        known_stats["right"] += 1
                    else:
                        known_stats["wrong"] += 1
                        if known_stats["first_failure"] is None:
                            known_stats["first_failure"] = {"split": split, "index": index, "kind": "wrong"}
                except Exception as exc:  # noqa: BLE001
                    known_stats["errors"] += 1
                    if known_stats["first_failure"] is None:
                        known_stats["first_failure"] = {"split": split, "index": index, "error": repr(exc)}
        streams: dict[str, object] = {}
        for seed in (31_000_000 + task, 32_000_000 + task):
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            stats = {"seed": seed, "right": 0, "wrong": 0, "errors": 0, "first_failure": None}
            valid = 0
            while valid < args.fresh:
                try:
                    example = generator.generate()
                    actual = rule(copy.deepcopy(example["input"]))
                    valid += 1
                    if same(actual, example["output"]):
                        stats["right"] += 1
                    else:
                        stats["wrong"] += 1
                        if stats["first_failure"] is None:
                            stats["first_failure"] = {"valid_case": valid, "kind": "wrong"}
                except Exception as exc:  # noqa: BLE001
                    stats["errors"] += 1
                    if stats["first_failure"] is None:
                        stats["first_failure"] = {"valid_case": valid + 1, "error": repr(exc)}
                    if stats["errors"] > args.fresh:
                        break
            streams[str(seed)] = stats
        output["tasks"][f"{task:03d}"] = {"generator_hash": mapping[f"{task:03d}"], "known": known_stats, "fresh": streams}
        audit_path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"task{task:03d} known={known_stats['right']}/{known_stats['right'] + known_stats['wrong']} fresh=" + ",".join(f"{s['right']}/{args.fresh}" for s in streams.values()), flush=True)


if __name__ == "__main__":
    main()
