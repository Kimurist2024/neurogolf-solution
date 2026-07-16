#!/usr/bin/env python3
"""Measure compact generator-derived rule candidates before ONNX translation."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(TASK_DIR))
common = importlib.import_module("common")


HASHES = {23: "150deff5", 187: "7b6016b9", 367: "e73095fd"}


def bounded_cross(grid: list[list[int]], fill: int, background: int) -> list[list[int]]:
    x = np.asarray(grid, dtype=np.uint8)
    fg = x != 0
    left = np.maximum.accumulate(fg, axis=1)
    right = np.maximum.accumulate(fg[:, ::-1], axis=1)[:, ::-1]
    top = np.maximum.accumulate(fg, axis=0)
    bottom = np.maximum.accumulate(fg[::-1], axis=0)[::-1]
    inside = (x == 0) & left & right & top & bottom
    out = np.where(x != 0, x, np.where(inside, fill, background))
    return out.tolist()


def local23(grid: list[list[int]], variant: int) -> list[list[int]]:
    x = np.asarray(grid, dtype=np.uint8) != 0
    h, w = x.shape
    p = np.pad(x, 1)
    # Every true 2x2 placement, expanded to all four member cells.
    anchors = p[1 : h + 1, 1 : w + 1] & p[1 : h + 1, 2 : w + 2]
    anchors &= p[2 : h + 2, 1 : w + 1] & p[2 : h + 2, 2 : w + 2]
    box = np.zeros_like(x)
    for dr in (0, 1):
        for dc in (0, 1):
            rs = slice(dr, min(h, h + dr))
            cs = slice(dc, min(w, w + dc))
            box[rs, cs] |= anchors[: h - dr, : w - dc]
    if variant == 1:
        # A square cell also needs at least one diagonal occupied neighbour.
        diag = (
            p[:h, :w]
            | p[:h, 2 : w + 2]
            | p[2 : h + 2, :w]
            | p[2 : h + 2, 2 : w + 2]
        )
        box &= diag
    out = np.zeros((h, w), dtype=np.uint8)
    out[x] = 2
    out[box] = 8
    return out.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20_000)
    args = parser.parse_args()
    count = args.count
    report = {}
    for task in HASHES:
        module = importlib.import_module(f"task_{HASHES[task]}")
        rows = {}
        for variant in ((0, 1) if task == 23 else (0,)):
            failures = []
            failure_count = 0
            for index in range(count):
                seed = 91_000_000 + task * count + index
                random.seed(seed)
                common.random.seed(seed)
                example = module.generate()
                if task == 23:
                    actual = local23(example["input"], variant)
                elif task == 187:
                    actual = bounded_cross(example["input"], 2, 3)
                else:
                    actual = bounded_cross(example["input"], 4, 0)
                if actual != example["output"]:
                    failure_count += 1
                    if len(failures) < 3:
                        failures.append(
                            {
                                "index": index,
                                "seed": seed,
                                "input": example["input"],
                                "expected": example["output"],
                                "actual": actual,
                            }
                        )
            rows[str(variant)] = {
                "total": count,
                "failures": failure_count,
                "first_failures": failures,
            }
            print(
                f"task{task:03d} variant={variant} "
                f"failures={failure_count}"
            )
        report[str(task)] = rows
    (HERE / "numpy_rule_probe.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
