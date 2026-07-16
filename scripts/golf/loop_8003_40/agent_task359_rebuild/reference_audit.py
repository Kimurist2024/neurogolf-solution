#!/usr/bin/env python3
"""Decode and verify task359's true rule on all known and fresh cases."""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARC_GEN = ROOT / "inputs/arc-gen-repo"
sys.path.insert(0, str(ARC_GEN))
generator = importlib.import_module("tasks.task_e26a3af2")


def exact_python_rule(grid: list[list[int]]) -> list[list[int]]:
    """Readable expansion of raw/task359.py, including Python tie-breaking."""
    columns = list(zip(*grid))
    output: list[list[int]] = []
    for row in grid:
        output_row = []
        for column in columns:
            sequence = row + list(column)
            output_row.append(max(sequence, key=sequence.count))
        output.append(output_row)
    return output


def numpy_reference(grid: list[list[int]]) -> list[list[int]]:
    """Histogram implementation with the exact first-occurrence tie break."""
    array = np.asarray(grid, dtype=np.int64)
    height, width = array.shape
    row_counts = np.stack(
        [(array == color).sum(axis=1) for color in range(10)], axis=1
    )
    col_counts = np.stack(
        [(array == color).sum(axis=0) for color in range(10)], axis=1
    )
    output = np.empty((height, width), dtype=np.int64)
    for row_index in range(height):
        for col_index in range(width):
            counts = row_counts[row_index] + col_counts[col_index]
            maximum = int(counts.max())
            # raw p iterates row first and then column; max returns the first
            # sequence element whose count reaches the maximum.
            sequence = np.concatenate((array[row_index], array[:, col_index]))
            output[row_index, col_index] = next(
                int(color) for color in sequence if counts[int(color)] == maximum
            )
    return output.tolist()


def generator_reference(grid: list[list[int]]) -> list[list[int]]:
    """Recover the horizontal or vertical stripes emitted by the generator.

    The clean orientation has 13--15 samples per stripe line and therefore a
    much larger agreement score than filling every line in the wrong
    orientation.  Ties within a line use the first-occurrence convention.
    """
    array = np.asarray(grid, dtype=np.int64)
    height, width = array.shape

    row_modes = []
    for row in array:
        values = row.tolist()
        row_modes.append(max(values, key=values.count))
    horizontal = np.repeat(np.asarray(row_modes)[:, None], width, axis=1)

    col_modes = []
    for column in array.T:
        values = column.tolist()
        col_modes.append(max(values, key=values.count))
    vertical = np.repeat(np.asarray(col_modes)[None, :], height, axis=0)

    horizontal_agreement = int(np.count_nonzero(horizontal == array))
    vertical_agreement = int(np.count_nonzero(vertical == array))
    # No tie occurred in the audited generator domain.  Horizontal is chosen
    # deterministically if one ever appears.
    return (horizontal if horizontal_agreement >= vertical_agreement else vertical).tolist()


def raw_mismatch_diagnostic(
    grid: list[list[int]], raw_output: list[list[int]], expected: list[list[int]]
) -> dict[str, object]:
    array = np.asarray(grid, dtype=np.int64)
    raw = np.asarray(raw_output, dtype=np.int64)
    target = np.asarray(expected, dtype=np.int64)
    mismatch = np.argwhere(raw != target)
    row_index, col_index = (int(value) for value in mismatch[0])
    row = array[row_index].tolist()
    column = array[:, col_index].tolist()
    sequence = row + column
    counts = [sequence.count(color) for color in range(10)]
    return {
        "different_cells": int(mismatch.shape[0]),
        "first_cell": [row_index, col_index],
        "raw_color": int(raw[row_index, col_index]),
        "generator_color": int(target[row_index, col_index]),
        "row": row,
        "column": column,
        "row_then_column_counts_by_color": counts,
        "max_count": max(counts),
        "first_max_color_in_sequence": next(
            int(color) for color in sequence if counts[int(color)] == max(counts)
        ),
        "explanation": "the exact raw row+column mode is displaced by another stripe/noise distribution; this is not a tie-break discrepancy",
    }


def main() -> None:
    data = json.loads((ROOT / "inputs/neurogolf-2026/task359.json").read_text())
    known = {
        "total": 0,
        "python_rule_matches_expected": 0,
        "numpy_matches_python_rule": 0,
        "generator_reference_matches_expected": 0,
        "failures": [],
    }
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(data[split]):
            known["total"] += 1
            python_output = exact_python_rule(example["input"])
            numpy_output = numpy_reference(example["input"])
            generator_output = generator_reference(example["input"])
            expected = example["output"]
            python_ok = python_output == expected
            numpy_ok = numpy_output == python_output
            generator_ok = generator_output == expected
            known["python_rule_matches_expected"] += int(python_ok)
            known["numpy_matches_python_rule"] += int(numpy_ok)
            known["generator_reference_matches_expected"] += int(generator_ok)
            if not (python_ok and numpy_ok and generator_ok):
                known["failures"].append(
                    {
                        "split": split,
                        "index": index,
                        "python_ok": python_ok,
                        "numpy_ok": numpy_ok,
                        "generator_ok": generator_ok,
                    }
                )

    random.seed(359_800_340)
    fresh = {
        "seed": 359_800_340,
        "total": 5000,
        "generator_errors": 0,
        "python_rule_matches_generated_output": 0,
        "numpy_matches_python_rule": 0,
        "generator_reference_matches_generated_output": 0,
        "failures": [],
        "shape_histogram": {},
    }
    for index in range(fresh["total"]):
        try:
            example = generator.generate()
            grid = example["input"]
            shape = f"{len(grid)}x{len(grid[0])}"
            fresh["shape_histogram"][shape] = fresh["shape_histogram"].get(shape, 0) + 1
            python_output = exact_python_rule(grid)
            numpy_output = numpy_reference(grid)
            generator_output = generator_reference(grid)
            python_ok = python_output == example["output"]
            numpy_ok = numpy_output == python_output
            generator_ok = generator_output == example["output"]
            fresh["python_rule_matches_generated_output"] += int(python_ok)
            fresh["numpy_matches_python_rule"] += int(numpy_ok)
            fresh["generator_reference_matches_generated_output"] += int(generator_ok)
            if not (python_ok and numpy_ok and generator_ok) and len(fresh["failures"]) < 20:
                failure = {
                    "index": index,
                    "shape": shape,
                    "python_ok": python_ok,
                    "numpy_ok": numpy_ok,
                    "generator_ok": generator_ok,
                }
                if not python_ok:
                    failure["raw_mismatch"] = raw_mismatch_diagnostic(
                        grid, python_output, example["output"]
                    )
                fresh["failures"].append(failure)
        except Exception as exc:  # noqa: BLE001
            fresh["generator_errors"] += 1
            if len(fresh["failures"]) < 20:
                fresh["failures"].append({"index": index, "error": repr(exc)})

    report = {
        "task": 359,
        "raw_rule": "for every (r,c), return the mode of row r concatenated with column c; ties choose the first occurrence in that row-then-column sequence",
        "pseudocode": [
            "columns = transpose(grid)",
            "for each row r and column c: S = row[r] + columns[c]",
            "count each color in S",
            "output[r,c] = first value in S having the maximum count",
        ],
        "generator_rule": "choose horizontal-vs-vertical stripe reconstruction by total line-mode agreement, then fill each selected line with its first-tied mode",
        "classification": "global row/column histogram, data-dependent orientation, argmax, and first-occurrence tie break",
        "known": known,
        "fresh": fresh,
        "reference_proved": bool(
            known["python_rule_matches_expected"] == known["total"] == 266
            and known["numpy_matches_python_rule"] == known["total"]
            and known["generator_reference_matches_expected"] == known["total"]
            and fresh["generator_errors"] == 0
            and fresh["numpy_matches_python_rule"] == fresh["total"]
            and fresh["generator_reference_matches_generated_output"] == fresh["total"]
        ),
    }
    (HERE / "REFERENCE_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "known": known,
        "fresh": {key: value for key, value in fresh.items() if key != "shape_histogram"},
        "reference_proved": report["reference_proved"],
    }, indent=2))


if __name__ == "__main__":
    main()
