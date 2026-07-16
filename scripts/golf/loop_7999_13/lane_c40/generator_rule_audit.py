#!/usr/bin/env python3
"""Prove and exercise the default-generator rule for task391/f8b3ba0a."""

from __future__ import annotations

import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))


def solve(example: dict[str, Any]) -> list[list[int]]:
    rendered = Counter(
        int(color)
        for row in example["input"]
        for color in row
        if int(color) != 0
    )
    ordered = sorted(rendered.items(), key=lambda item: item[1], reverse=True)
    return [[color] for color, _count in ordered[1:4]]


def facts(example: dict[str, Any]) -> dict[str, Any]:
    rendered = Counter(
        int(color)
        for row in example["input"]
        for color in row
        if int(color) != 0
    )
    counts = sorted(rendered.values(), reverse=True)
    return {
        "colors": len(rendered),
        "rendered_counts_desc": counts,
        "four_colors": len(rendered) == 4,
        "minority_counts_distinct": len(set(counts[1:])) == 3,
        "minority_counts_in_2_4_6_8": set(counts[1:]).issubset({2, 4, 6, 8}),
        "background_strictly_largest": counts[0] > counts[1],
    }


def main() -> int:
    generator = importlib.import_module("task_f8b3ba0a")
    known = json.loads((ROOT / "inputs/neurogolf-2026/task391.json").read_text())
    known_right = known_wrong = 0
    first_known_wrong = None
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(known[subset]):
            if solve(example) == example["output"]:
                known_right += 1
            else:
                known_wrong += 1
                first_known_wrong = first_known_wrong or {"subset": subset, "index": index}

    seed = 391_800_046
    random.seed(seed)
    fresh_right = fresh_wrong = generation_errors = 0
    first_fresh_wrong = None
    fact_failures: Counter[str] = Counter()
    minority_sets: Counter[str] = Counter()
    background_min = None
    background_max = None
    for index in range(5000):
        try:
            example = generator.generate()
        except Exception:  # noqa: BLE001 - evidence counter
            generation_errors += 1
            continue
        row = facts(example)
        for key in (
            "four_colors",
            "minority_counts_distinct",
            "minority_counts_in_2_4_6_8",
            "background_strictly_largest",
        ):
            if not row[key]:
                fact_failures[key] += 1
        counts = row["rendered_counts_desc"]
        background_min = counts[0] if background_min is None else min(background_min, counts[0])
        background_max = counts[0] if background_max is None else max(background_max, counts[0])
        minority_sets[str(sorted(counts[1:]))] += 1
        if solve(example) == example["output"]:
            fresh_right += 1
        else:
            fresh_wrong += 1
            first_fresh_wrong = first_fresh_wrong or {"index": index, "facts": row}

    report = {
        "task": 391,
        "generator_hash": "f8b3ba0a",
        "rule": (
            "Count nonzero rendered colors, remove the strictly most frequent background, "
            "and emit the remaining three colors by descending frequency."
        ),
        "static_proof": {
            "logical_dimensions": {"width": [3, 5], "height": [6, 7]},
            "minority_logical_counts": "three values sampled without replacement from {1,2,3,4}",
            "minority_rendered_counts": "three distinct values from {2,4,6,8}",
            "minimum_background_logical_count": 3 * 6 - (4 + 3 + 2),
            "maximum_minority_logical_count": 4,
            "background_is_strictly_largest": True,
            "tie_warning_resolution": (
                "The public generate(width, colors) override can accept tied arbitrary colors, "
                "but default/private generation cannot tie because counts are sampled without replacement."
            ),
        },
        "known": {
            "right": known_right,
            "wrong": known_wrong,
            "first_wrong": first_known_wrong,
        },
        "fresh": {
            "seed": seed,
            "requested": 5000,
            "right": fresh_right,
            "wrong": fresh_wrong,
            "generation_errors": generation_errors,
            "first_wrong": first_fresh_wrong,
            "fact_failures": dict(fact_failures),
            "minority_rendered_count_sets": dict(minority_sets),
            "background_rendered_count_range": [background_min, background_max],
        },
        "pass": known_wrong == 0
        and fresh_wrong == 0
        and generation_errors == 0
        and not fact_failures,
    }
    (HERE / "generator_rule_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
