#!/usr/bin/env python3
"""Disprove the tempting task310 ``count < 29`` selector simplification."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
import task_c909285e as generator  # noqa: E402


def main() -> None:
    count = 50_000
    seed = 20260714
    random.seed(seed)
    gaps: Counter[int] = Counter()
    equivalent = 0
    for _ in range(count):
        grid = np.asarray(generator.generate()["input"])
        counts = np.bincount(grid.ravel(), minlength=10)
        incumbent = (counts < 25) | (counts == 28)
        tempting = counts < 29
        equivalent += int(np.array_equal(incumbent, tempting))
        gaps.update(int(value) for value in counts[(counts >= 25) & (counts <= 27)])
    result = {
        "task": 310,
        "count": count,
        "seed": seed,
        "incumbent_vs_less29_equal_cases": equivalent,
        "different_cases": count - equivalent,
        "legal_gap_counts": dict(sorted(gaps.items())),
        "less29_domain_exact": equivalent == count,
    }
    (HERE / "task310_count_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
